/**
 * Tennis Media Worker
 *
 * Serves the highlights gallery and handles video uploads.
 * Media files stored in R2 under highlights/ prefix, served at root URLs.
 *
 * Routes:
 *   GET  /                         → gallery index
 *   GET  /{vid}/{file}.mp4         → video (falls back to highlights/{vid}/ in R2)
 *   GET  /thumbs/{vid}.jpg         → thumbnail (falls back to highlights/thumbs/)
 *   GET  /highlights/*             → backward compat (direct R2 key)
 *   POST /api/upload/init          → start upload
 *   PUT  /api/upload/:id/:part     → upload chunk
 *   POST /api/upload/:id/complete  → finalize upload
 *   POST /api/upload/link          → iCloud share link
 *   POST /api/upload/iphone        → iOS Shortcut single-shot upload (Bearer auth)
 *                                    (CF edge has a 100 MB body cap — files
 *                                     larger than that must use the chunked
 *                                     endpoints below)
 *   GET  /api/upload/iphone/check  → idempotency probe (?asset_id=…)
 *   POST /api/upload/iphone/init           → start chunked R2 upload (Bearer)
 *   PUT  /api/upload/iphone/:upid/:part    → upload one chunk
 *   POST /api/upload/iphone/:upid/complete → finalize + write marker
 *   GET  /api/status/:id           → check processing status
 *   POST /api/status/:id/update    → update processing status (auth)
 *   GET  /api/queue                → list all uploads and their status
 *
 *   POST /api/auth/apple           → exchange Apple identity token → our JWT (PR 1)
 *   GET  /api/me                   → current user's profile (Bearer JWT)
 *   DELETE /api/account            → tombstone + delete user's videos (PR 6)
 *
 *   GET  /u/{hash}                 → per-user gallery, cookie-auth (PR per-user)
 *   GET  /u/{hash}/{path}          → per-user asset (videos, thumbs, json)
 *
 *   GET  /privacy                  → static privacy policy (PR 6)
 *
 *   playfullife.com/*              → redirect to tennis.playfullife.com
 *   media.playfullife.com/*        → redirect to tennis.playfullife.com
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Redirect bare domain and legacy media subdomain to tennis subdomain
    if (url.hostname === 'playfullife.com' || url.hostname === 'www.playfullife.com' || url.hostname === 'media.playfullife.com') {
      return Response.redirect(
        `https://tennis.playfullife.com${url.pathname}${url.search}`,
        301
      );
    }

    const path = url.pathname;

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders() });
    }

    // API routes
    if (path.startsWith('/api/')) {
      return handleApi(request, env, path);
    }

    // Per-user gallery: /u/<hash> and /u/<hash>/<rest> — cookie-auth gated.
    // Match before generic asset handling so the auth gate always runs.
    const userMatch = path.match(/^\/u\/(u_[a-f0-9]{8})(\/.*)?$/);
    if (userMatch && (request.method === 'GET' || request.method === 'HEAD')) {
      return handleUserAsset(request, env, userMatch[1], userMatch[2] || '/');
    }

    // Static assets from R2
    if (request.method === 'GET' || request.method === 'HEAD') {
      return handleAsset(request, env, path);
    }

    return new Response('Method Not Allowed', { status: 405 });
  },
};

// ---------------------------------------------------------------------------
// Asset serving from R2
// ---------------------------------------------------------------------------

async function handleAsset(request, env, path) {
  // Resolve R2 key — root serves the gallery index
  let key;
  if (path === '/' || path === '/index.html') {
    key = 'highlights/index.html';
  } else if (path === '/privacy' || path === '/privacy.html') {
    key = 'static/privacy.html';
  } else if (path === '/support' || path === '/support.html') {
    key = 'static/support.html';
  } else {
    key = path.slice(1); // strip leading /
  }

  if (request.method === 'HEAD') {
    return handleHead(env, key);
  }

  return serveR2Object(request, env, key, { fallbackHighlightsPrefix: true });
}

// Serve an R2 object as an HTTP response. Handles range requests, ?dl=1,
// content-type-aware caching, and an optional fallback to "highlights/" +
// key (used by the legacy flat layout). Caller is responsible for auth.
async function serveR2Object(request, env, key, opts = {}) {
  const reqUrl = new URL(request.url);
  const isDownload = reqUrl.searchParams.get('dl') === '1';
  const rangeHeader = isDownload ? null : request.headers.get('range');
  const r2Range = parseRange(rangeHeader);
  const getOpts = {};
  if (r2Range) getOpts.range = r2Range;

  let obj = null;
  try {
    obj = await env.BUCKET.get(key, getOpts);
  } catch {}

  if (!obj && opts.fallbackHighlightsPrefix &&
      !key.startsWith('highlights/') && !key.startsWith('uploads/')) {
    try {
      obj = await env.BUCKET.get('highlights/' + key, getOpts);
    } catch {}
  }

  if (!obj) {
    return new Response('Not Found', { status: 404 });
  }

  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  headers.set('etag', obj.httpEtag);
  headers.set('accept-ranges', 'bytes');
  headers.set('access-control-allow-origin', '*');
  if (opts.extraHeaders) {
    for (const [k, v] of Object.entries(opts.extraHeaders)) headers.set(k, v);
  }

  if (isDownload) {
    const filename = key.split('/').pop();
    headers.set('content-disposition', `attachment; filename="${filename}"`);
  }

  const ct = (headers.get('content-type') || '').toLowerCase();
  const isHtml = ct.includes('html') || key.endsWith('.html') || key === 'highlights/';
  if (ct.startsWith('video/')) {
    headers.set('cache-control', 'public, max-age=86400');
  } else if (isHtml) {
    headers.set('cache-control', 'no-store, no-cache, must-revalidate, max-age=0');
    headers.set('cdn-cache-control', 'no-store');
  } else {
    headers.set('cache-control', 'public, max-age=3600');
  }

  if (!obj.body) {
    return new Response(null, { status: 304, headers });
  }

  if (obj.range && rangeHeader && !isDownload) {
    const { offset, length } = obj.range;
    headers.set('content-range', `bytes ${offset}-${offset + length - 1}/${obj.size}`);
    headers.set('content-length', String(length));
    return new Response(obj.body, { status: 206, headers });
  }

  headers.set('content-length', String(obj.size));
  if (isHtml) {
    headers.set('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
    headers.set('Pragma', 'no-cache');
    headers.set('Expires', '0');
  }
  return new Response(obj.body, { status: 200, headers });
}

// ---------------------------------------------------------------------------
// Per-user gallery — /u/<hash>/...
// ---------------------------------------------------------------------------
//
// Auth: either a `tennis_jwt` cookie or a `?t=<jwt>` query param. The query
// param path is the iOS WebView's bootstrap — on success we redirect to the
// clean URL with Set-Cookie so subresources (video src, img src) load with
// the cookie alone. Admins (env.ADMIN_USER_HASHES) can view any user's URL.
async function handleUserAsset(request, env, userHash, subpath) {
  const url = new URL(request.url);
  const queryToken = url.searchParams.get('t');
  const cookieToken = readCookie(request, 'tennis_jwt');
  const token = queryToken || cookieToken;

  let claims = null;
  if (token && env.JWT_SIGNING_SECRET) {
    try { claims = await verifyOurJWT(token, env.JWT_SIGNING_SECRET); } catch {}
  }
  if (!claims) {
    return new Response(
      'Sign-in required. Open this gallery from the Tennis Uploader iOS app.',
      { status: 401, headers: { 'content-type': 'text/plain; charset=utf-8' } },
    );
  }
  if (claims.sub !== userHash && !isAdminUser(env, claims.sub)) {
    return new Response('Forbidden', { status: 403 });
  }

  // Bootstrap: ?t=<jwt> → Set-Cookie + 302 to clean URL.
  if (queryToken) {
    url.searchParams.delete('t');
    const cleanUrl = url.pathname + (url.searchParams.toString() ? '?' + url.searchParams.toString() : '');
    return new Response(null, {
      status: 302,
      headers: {
        'location': cleanUrl,
        // 30 days; Secure required for SameSite=None but we use Lax (same-site).
        'set-cookie': `tennis_jwt=${queryToken}; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age=2592000`,
      },
    });
  }

  // Resolve R2 key under highlights/<user_hash>/...
  let key;
  if (subpath === '/' || subpath === '') {
    key = `highlights/${userHash}/index.html`;
  } else {
    key = `highlights/${userHash}${subpath}`;
  }

  if (request.method === 'HEAD') {
    return handleHead(env, key);
  }
  return serveR2Object(request, env, key);
}

function readCookie(request, name) {
  const cookieHeader = request.headers.get('cookie') || '';
  for (const c of cookieHeader.split(';')) {
    const [k, ...rest] = c.trim().split('=');
    if (k === name) return rest.join('=');
  }
  return null;
}

function isAdminUser(env, userHash) {
  const raw = env.ADMIN_USER_HASHES || '';
  return raw.split(',').map((s) => s.trim()).filter(Boolean).includes(userHash);
}

async function handleHead(env, key) {
  let obj = await env.BUCKET.head(key);
  if (!obj && !key.startsWith('highlights/') && !key.startsWith('uploads/')) {
    obj = await env.BUCKET.head('highlights/' + key);
  }
  if (!obj) {
    return new Response(null, { status: 404 });
  }

  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  headers.set('etag', obj.httpEtag);
  headers.set('content-length', String(obj.size));
  headers.set('accept-ranges', 'bytes');
  headers.set('access-control-allow-origin', '*');
  return new Response(null, { status: 200, headers });
}

// ---------------------------------------------------------------------------
// API routes
// ---------------------------------------------------------------------------

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

async function handleApi(request, env, path) {
  const cors = corsHeaders();

  try {
    // Auth — Sign in with Apple (PR 1)
    if (path === '/api/auth/apple' && request.method === 'POST') {
      return await handleAuthApple(request, env, cors);
    }

    if (path === '/api/me' && request.method === 'GET') {
      return await handleMe(request, env, cors);
    }

    if (path === '/api/account' && request.method === 'DELETE') {
      return await handleDeleteAccount(request, env, cors);
    }

    if (path === '/api/upload/init' && request.method === 'POST') {
      return await handleInit(request, env, cors);
    }

    const partMatch = path.match(/^\/api\/upload\/([^/]+)\/(\d+)$/);
    if (partMatch && request.method === 'PUT') {
      return await handlePart(
        request,
        env,
        cors,
        partMatch[1],
        parseInt(partMatch[2])
      );
    }

    if (path === '/api/upload/link' && request.method === 'POST') {
      return await handleLink(request, env, cors);
    }

    if (path === '/api/upload/iphone' && request.method === 'POST') {
      return await handleIphoneUpload(request, env, cors);
    }

    if (path === '/api/upload/iphone/check' && request.method === 'GET') {
      return await handleIphoneCheck(request, env, cors);
    }

    if (path === '/api/upload/iphone/init' && request.method === 'POST') {
      return await handleIphoneInit(request, env, cors);
    }

    const ipPartMatch = path.match(/^\/api\/upload\/iphone\/([^/]+)\/(\d+)$/);
    if (ipPartMatch && request.method === 'PUT') {
      return await handleIphonePart(
        request,
        env,
        cors,
        ipPartMatch[1],
        parseInt(ipPartMatch[2]),
      );
    }

    const ipCompleteMatch = path.match(/^\/api\/upload\/iphone\/([^/]+)\/complete$/);
    if (ipCompleteMatch && request.method === 'POST') {
      return await handleIphoneComplete(request, env, cors, ipCompleteMatch[1]);
    }

    const completeMatch = path.match(/^\/api\/upload\/([^/]+)\/complete$/);
    if (completeMatch && request.method === 'POST') {
      return await handleComplete(request, env, cors, completeMatch[1]);
    }

    const statusMatch = path.match(/^\/api\/status\/([^/]+)$/);
    if (statusMatch && request.method === 'GET') {
      return await handleStatus(env, cors, statusMatch[1]);
    }

    const updateMatch = path.match(/^\/api\/status\/([^/]+)\/update$/);
    if (updateMatch && request.method === 'POST') {
      return await handleStatusUpdate(request, env, cors, updateMatch[1]);
    }

    if (path === '/api/queue' && request.method === 'GET') {
      return await handleQueue(env, cors);
    }

    if (path === '/api/tags' && request.method === 'GET') {
      return await handleGetTags(env, cors);
    }

    if (path === '/api/tags' && request.method === 'POST') {
      return await handleSetTags(request, env, cors);
    }

    const deleteMatch = path.match(/^\/api\/video\/([^/]+)\/delete$/);
    if (deleteMatch && request.method === 'POST') {
      return await handleDeleteVideo(request, env, cors, deleteMatch[1]);
    }

    return jsonResponse({ error: 'Not found' }, 404, cors);
  } catch (err) {
    return jsonResponse({ error: err.message }, 500, cors);
  }
}

// ---------------------------------------------------------------------------
// Upload handlers
// ---------------------------------------------------------------------------

async function handleInit(request, env, cors) {
  const body = await request.json();
  const { password, filename } = body;

  if (!password || password !== env.UPLOAD_PASSWORD) {
    return jsonResponse({ error: 'Invalid password' }, 403, cors);
  }

  if (!filename) {
    return jsonResponse({ error: 'filename required' }, 400, cors);
  }

  const ext = filename.split('.').pop().toLowerCase();
  if (!['mov', 'mp4'].includes(ext)) {
    return jsonResponse({ error: 'Only .mov and .mp4 files allowed' }, 400, cors);
  }

  const id = generateId();
  const key = `uploads/${id}.${ext}`;

  const multipart = await env.BUCKET.createMultipartUpload(key);

  const metadata = {
    id,
    filename,
    uploaded_at: new Date().toISOString(),
    status: 'uploading',
    key,
    uploadId: multipart.uploadId,
  };
  await env.BUCKET.put(`uploads/${id}.json`, JSON.stringify(metadata), {
    httpMetadata: { contentType: 'application/json' },
  });

  return jsonResponse({ id, uploadId: multipart.uploadId, key }, 200, cors);
}

async function handlePart(request, env, cors, id, partNumber) {
  const metaObj = await env.BUCKET.get(`uploads/${id}.json`);
  if (!metaObj) {
    return jsonResponse({ error: 'Upload not found' }, 404, cors);
  }
  const meta = await metaObj.json();

  if (!meta.uploadId) {
    return jsonResponse({ error: 'Upload already completed' }, 400, cors);
  }

  const upload = env.BUCKET.resumeMultipartUpload(meta.key, meta.uploadId);
  const part = await upload.uploadPart(partNumber, request.body);

  return jsonResponse(
    { partNumber: part.partNumber, etag: part.etag },
    200,
    cors
  );
}

async function handleComplete(request, env, cors, id) {
  const metaObj = await env.BUCKET.get(`uploads/${id}.json`);
  if (!metaObj) {
    return jsonResponse({ error: 'Upload not found' }, 404, cors);
  }
  const meta = await metaObj.json();

  if (!meta.uploadId) {
    return jsonResponse({ error: 'Upload already completed' }, 400, cors);
  }

  const body = await request.json();
  const { parts } = body;

  const upload = env.BUCKET.resumeMultipartUpload(meta.key, meta.uploadId);
  await upload.complete(
    parts.map((p) => ({ partNumber: p.partNumber, etag: p.etag }))
  );

  meta.status = 'pending';
  meta.completed_at = new Date().toISOString();
  delete meta.uploadId;
  await env.BUCKET.put(`uploads/${id}.json`, JSON.stringify(meta), {
    httpMetadata: { contentType: 'application/json' },
  });

  return jsonResponse({ id, status: 'pending' }, 200, cors);
}

async function handleLink(request, env, cors) {
  const body = await request.json();
  const { password, url } = body;

  if (!password || password !== env.UPLOAD_PASSWORD) {
    return jsonResponse({ error: 'Invalid password' }, 403, cors);
  }

  if (!url || !url.includes('icloud.com/')) {
    return jsonResponse({ error: 'Valid iCloud share link required' }, 400, cors);
  }

  const id = generateId();
  const metadata = {
    id,
    type: 'icloud_link',
    url,
    uploaded_at: new Date().toISOString(),
    status: 'pending',
  };
  await env.BUCKET.put(`uploads/${id}.json`, JSON.stringify(metadata), {
    httpMetadata: { contentType: 'application/json' },
  });

  return jsonResponse({ id, status: 'pending' }, 200, cors);
}

// ---------------------------------------------------------------------------
// iPhone Shortcut upload — single-shot streaming POST with Bearer auth
//
// Headers (required unless noted):
//   Authorization: Bearer <IPHONE_UPLOAD_TOKEN>
//   X-Asset-Id:    iOS PHAsset localIdentifier (idempotency key)
//   X-Filename:    original filename, e.g. IMG_1234.MOV
//   X-Created-At:  ISO-8601 recording date (optional)
//
// Body: raw video bytes (NOT multipart). Apple Shortcuts can post raw via
//   "Get URL Contents" with body type "File". Multipart is harder to assemble.
//
// Responses:
//   200: { video_id, status: 'queued', r2_key, asset_id }
//   401: missing/invalid bearer
//   400: missing required header
//   409: duplicate (asset_id already uploaded) — returns existing video_id
//   500: storage error
// ---------------------------------------------------------------------------

async function handleIphoneUpload(request, env, cors) {
  const auth = request.headers.get('authorization') || '';
  const expected = env.IPHONE_UPLOAD_TOKEN;
  if (!expected || auth !== `Bearer ${expected}`) {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }

  const assetId = request.headers.get('x-asset-id');
  const filename = request.headers.get('x-filename');
  const createdAt = request.headers.get('x-created-at') || '';

  if (!assetId || !filename) {
    return jsonResponse({ error: 'X-Asset-Id and X-Filename headers required' }, 400, cors);
  }

  if (!request.body) {
    return jsonResponse({ error: 'request body required' }, 400, cors);
  }

  // Sanitize filename: extension only matters for content-type; keep alphanumeric core.
  const ext = (filename.split('.').pop() || 'mov').toLowerCase();
  if (!['mov', 'mp4'].includes(ext)) {
    return jsonResponse({ error: 'Only .mov and .mp4 files allowed' }, 400, cors);
  }

  // Deterministic video_id from asset_id so retries are idempotent.
  const videoId = await iphoneVideoIdFromAssetId(assetId);
  const r2Key = `source/${videoId}.${ext}`;

  // Idempotency check: if R2 already has this key, treat as duplicate.
  try {
    const existing = await env.BUCKET.head(r2Key);
    if (existing) {
      return jsonResponse(
        {
          video_id: videoId,
          status: 'duplicate',
          r2_key: r2Key,
          asset_id: assetId,
          message: 'asset_id already uploaded',
        },
        409,
        cors,
      );
    }
  } catch {}

  // Stream the body directly to R2.
  try {
    await env.BUCKET.put(r2Key, request.body, {
      httpMetadata: {
        contentType: ext === 'mp4' ? 'video/mp4' : 'video/quicktime',
        contentDisposition: `attachment; filename="${filename}"`,
      },
      customMetadata: {
        ios_asset_id: assetId,
        original_filename: filename,
        created_at: createdAt,
        uploaded_at: new Date().toISOString(),
        source: 'iphone_shortcut',
      },
    });
  } catch (err) {
    return jsonResponse({ error: 'R2 put failed: ' + err.message }, 500, cors);
  }

  // Marker file picked up by the Hetzner-side poller, which creates the
  // coordinator job. This is the primary registration path; the HTTP fetch
  // below is an optional fast-track that requires a CF DNS-only subdomain
  // (CF blocks Worker fetches to bare origin IPs — error 1003).
  const markerKey = `uploads/${videoId}.json`;
  try {
    await env.BUCKET.put(
      markerKey,
      JSON.stringify({
        video_id: videoId,
        asset_id: assetId,
        filename,
        created_at: createdAt,
        uploaded_at: new Date().toISOString(),
        r2_source_key: r2Key,
        source: 'iphone_shortcut',
        status: 'awaiting_coordinator',
      }),
      { httpMetadata: { contentType: 'application/json' } },
    );
  } catch (err) {
    // Non-fatal: MOV is in R2; the user can re-run the Shortcut to
    // re-write the marker. Log via response field for diagnostic visibility.
    return jsonResponse(
      {
        video_id: videoId,
        status: 'queued_no_marker',
        r2_key: r2Key,
        asset_id: assetId,
        marker_error: err.message,
      },
      200,
      cors,
    );
  }

  // Register the job with the coordinator. Failures here are non-fatal for
  // the upload itself (the MOV is in R2) — the user can retry to re-trigger,
  // or a future poller can backfill from R2.
  let coordinatorStatus = 'skipped';
  let coordinatorError = null;
  if (env.COORDINATOR_URL && env.COORDINATOR_TOKEN) {
    try {
      const coordResp = await fetch(`${env.COORDINATOR_URL}/jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${env.COORDINATOR_TOKEN}`,
        },
        body: JSON.stringify({
          icloud_asset_id: assetId,
          filename,
          album_name: 'iphone_shortcut',
          video_id: videoId,
        }),
      });
      const rawText = await coordResp.text();
      let coordData = {};
      try { coordData = JSON.parse(rawText); } catch {}
      if (coordResp.ok) {
        coordinatorStatus = coordData.status || 'created';
      } else {
        coordinatorStatus = 'failed';
        coordinatorError = `HTTP ${coordResp.status}: ${rawText.slice(0, 200)}`;
      }
    } catch (err) {
      coordinatorStatus = 'failed';
      coordinatorError = err.message;
    }
  }

  return jsonResponse(
    {
      video_id: videoId,
      status: 'queued',
      r2_key: r2Key,
      asset_id: assetId,
      coordinator: coordinatorStatus,
      coordinator_error: coordinatorError,
    },
    200,
    cors,
  );
}

async function handleIphoneCheck(request, env, cors) {
  const auth = await authenticateIphoneUpload(request, env);
  if (!auth.ok) {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }

  const url = new URL(request.url);
  const assetId = url.searchParams.get('asset_id');
  if (!assetId) {
    return jsonResponse({ error: 'asset_id query param required' }, 400, cors);
  }

  const videoId = await iphoneVideoIdFromAssetId(assetId);
  for (const ext of ['mov', 'mp4']) {
    const r2Key = `source/${videoId}.${ext}`;
    const obj = await env.BUCKET.head(r2Key);
    if (obj) {
      return jsonResponse(
        {
          uploaded: true,
          video_id: videoId,
          r2_key: r2Key,
          size: obj.size,
        },
        200,
        cors,
      );
    }
  }

  return jsonResponse({ uploaded: false, video_id: videoId }, 200, cors);
}

// ---------------------------------------------------------------------------
// iPhone chunked upload — for files larger than CF edge's 100 MB body cap.
//
// Flow:
//   POST /api/upload/iphone/init      { asset_id, filename, ext }
//     → { upload_id, r2_key, video_id }
//   PUT  /api/upload/iphone/<id>/<n>  body = chunk bytes
//     → { partNumber, etag }   (must be ≥5 MB except last; ≤100 MB always)
//   POST /api/upload/iphone/<id>/complete   { parts: [{partNumber, etag}, …] }
//     → { video_id, status: 'queued', r2_key }
// ---------------------------------------------------------------------------

async function handleIphoneInit(request, env, cors) {
  const auth = await authenticateIphoneUpload(request, env);
  if (!auth.ok) {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }
  const body = await request.json().catch(() => ({}));
  const { asset_id: assetId, filename } = body;
  const createdAt = body.created_at || '';
  if (!assetId || !filename) {
    return jsonResponse({ error: 'asset_id and filename required' }, 400, cors);
  }
  const ext = (filename.split('.').pop() || 'mov').toLowerCase();
  if (!['mov', 'mp4'].includes(ext)) {
    return jsonResponse({ error: 'Only .mov and .mp4 allowed' }, 400, cors);
  }
  const videoId = await iphoneVideoIdFromAssetId(assetId);
  const r2Key = `source/${videoId}.${ext}`;

  // Idempotency: if the source key already exists, no need to re-upload.
  try {
    const existing = await env.BUCKET.head(r2Key);
    if (existing) {
      return jsonResponse(
        { video_id: videoId, status: 'duplicate', r2_key: r2Key, asset_id: assetId },
        409, cors,
      );
    }
  } catch {}

  const multipart = await env.BUCKET.createMultipartUpload(r2Key, {
    httpMetadata: {
      contentType: ext === 'mp4' ? 'video/mp4' : 'video/quicktime',
      contentDisposition: `attachment; filename="${filename}"`,
    },
    customMetadata: {
      ios_asset_id: assetId,
      original_filename: filename,
      created_at: createdAt,
      uploaded_at: new Date().toISOString(),
      source: 'iphone_shortcut',
    },
  });

  // Stash the in-flight upload metadata in R2 so subsequent part/complete
  // calls can resume without client-side state.
  const stateKey = `uploads/_inflight_${videoId}.json`;
  await env.BUCKET.put(
    stateKey,
    JSON.stringify({
      video_id: videoId,
      asset_id: assetId,
      filename,
      created_at: createdAt,
      r2_key: r2Key,
      ext,
      r2_upload_id: multipart.uploadId,
      created_at_inflight: new Date().toISOString(),
      uploaded_by: auth.kind === 'user' ? auth.user_hash : null,
    }),
    { httpMetadata: { contentType: 'application/json' } },
  );

  return jsonResponse(
    {
      video_id: videoId,
      upload_id: videoId,  // we use videoId as upload_id for client convenience
      r2_key: r2Key,
      asset_id: assetId,
    },
    200, cors,
  );
}

async function handleIphonePart(request, env, cors, uploadId, partNumber) {
  const auth = await authenticateIphoneUpload(request, env);
  if (!auth.ok) {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }

  const stateKey = `uploads/_inflight_${uploadId}.json`;
  const stateObj = await env.BUCKET.get(stateKey);
  if (!stateObj) {
    return jsonResponse({ error: 'Unknown or expired upload_id' }, 404, cors);
  }
  const state = await stateObj.json();

  const upload = env.BUCKET.resumeMultipartUpload(state.r2_key, state.r2_upload_id);
  const part = await upload.uploadPart(partNumber, request.body);
  return jsonResponse(
    { partNumber: part.partNumber, etag: part.etag },
    200, cors,
  );
}

async function handleIphoneComplete(request, env, cors, uploadId) {
  const auth = await authenticateIphoneUpload(request, env);
  if (!auth.ok) {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }

  const stateKey = `uploads/_inflight_${uploadId}.json`;
  const stateObj = await env.BUCKET.get(stateKey);
  if (!stateObj) {
    return jsonResponse({ error: 'Unknown or expired upload_id' }, 404, cors);
  }
  const state = await stateObj.json();

  const body = await request.json();
  const parts = body.parts;
  if (!Array.isArray(parts) || parts.length === 0) {
    return jsonResponse({ error: 'parts[] required' }, 400, cors);
  }

  const upload = env.BUCKET.resumeMultipartUpload(state.r2_key, state.r2_upload_id);
  await upload.complete(
    parts.map((p) => ({ partNumber: p.partNumber, etag: p.etag })),
  );

  // Write marker file so the Hetzner poller registers the job.
  const markerKey = `uploads/${state.video_id}.json`;
  await env.BUCKET.put(
    markerKey,
    JSON.stringify({
      video_id: state.video_id,
      asset_id: state.asset_id,
      filename: state.filename,
      created_at: state.created_at,
      uploaded_at: new Date().toISOString(),
      r2_source_key: state.r2_key,
      source: auth.kind === 'user' ? 'iphone_app' : 'iphone_shortcut',
      status: 'awaiting_coordinator',
      uploaded_by: state.uploaded_by || (auth.kind === 'user' ? auth.user_hash : null),
    }),
    { httpMetadata: { contentType: 'application/json' } },
  );

  // Bump user's video count on success (best-effort).
  if (auth.kind === 'user') {
    try {
      // We don't know the apple_sub from the JWT alone here without re-verifying,
      // but the JWT carries apple_sub in claims — fetch fresh via authenticateUser.
      const userResult = await authenticateUser(request, env);
      if (userResult.kind === 'user' && userResult.claims.apple_sub) {
        const userKey = `users/${userResult.claims.apple_sub}.json`;
        const userObj = await env.BUCKET.get(userKey);
        if (userObj) {
          const user = await userObj.json();
          user.video_count = (user.video_count || 0) + 1;
          user.last_upload_at = new Date().toISOString();
          await env.BUCKET.put(userKey, JSON.stringify(user), {
            httpMetadata: { contentType: 'application/json' },
          });
        }
      }
    } catch {}
  }

  // Clean up in-flight state.
  try { await env.BUCKET.delete(stateKey); } catch {}

  return jsonResponse(
    {
      video_id: state.video_id,
      status: 'queued',
      r2_key: state.r2_key,
      asset_id: state.asset_id,
    },
    200, cors,
  );
}

async function iphoneVideoIdFromAssetId(assetId) {
  const data = new TextEncoder().encode(assetId);
  const hash = await crypto.subtle.digest('SHA-256', data);
  const hex = [...new Uint8Array(hash)]
    .slice(0, 4)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  return `iphone_${hex}`;
}

async function handleStatus(env, cors, id) {
  const metaObj = await env.BUCKET.get(`uploads/${id}.json`);
  if (metaObj) {
    return jsonResponse(await metaObj.json(), 200, cors);
  }
  return jsonResponse({ id, status: 'not_found' }, 404, cors);
}

async function handleStatusUpdate(request, env, cors, id) {
  const body = await request.json();
  const { password, status, stage, progress, error, video_url } = body;

  if (!password || password !== env.UPLOAD_PASSWORD) {
    return jsonResponse({ error: 'Invalid password' }, 403, cors);
  }

  const metaObj = await env.BUCKET.get(`uploads/${id}.json`);
  if (!metaObj) {
    return jsonResponse({ error: 'Upload not found' }, 404, cors);
  }
  const meta = await metaObj.json();

  if (status) meta.status = status;
  if (stage) meta.stage = stage;
  if (progress !== undefined) meta.progress = progress;
  if (error) meta.error = error;
  if (video_url) meta.video_url = video_url;
  meta.updated_at = new Date().toISOString();

  await env.BUCKET.put(`uploads/${id}.json`, JSON.stringify(meta), {
    httpMetadata: { contentType: 'application/json' },
  });

  return jsonResponse({ id, status: meta.status }, 200, cors);
}

async function handleQueue(env, cors) {
  // List all upload metadata JSONs
  const listed = await env.BUCKET.list({ prefix: 'uploads/', delimiter: '/' });
  const items = [];

  for (const obj of listed.objects) {
    if (!obj.key.endsWith('.json')) continue;
    try {
      const metaObj = await env.BUCKET.get(obj.key);
      if (metaObj) {
        const meta = await metaObj.json();
        // Only include relevant fields (not uploadId or internal keys)
        items.push({
          id: meta.id,
          filename: meta.filename || meta.url || 'Unknown',
          status: meta.status,
          stage: meta.stage || null,
          progress: meta.progress || null,
          uploaded_at: meta.uploaded_at,
          updated_at: meta.updated_at || meta.completed_at || meta.uploaded_at,
          video_url: meta.video_url || null,
          error: meta.error || null,
        });
      }
    } catch {}
  }

  // Sort by upload time, newest first
  items.sort((a, b) => (b.uploaded_at || '').localeCompare(a.uploaded_at || ''));

  return jsonResponse({ queue: items }, 200, cors);
}

// ---------------------------------------------------------------------------
// Delete a video and all its files (videos, thumbnail, meta.json)
// ---------------------------------------------------------------------------

async function handleDeleteVideo(request, env, cors, vid) {
  const body = await request.json();
  const { password } = body;

  if (password !== 'deletevideo') {
    return jsonResponse({ error: 'Invalid delete password' }, 403, cors);
  }

  // Sanitize vid: only allow safe characters
  if (!/^[A-Za-z0-9_-]+$/.test(vid)) {
    return jsonResponse({ error: 'Invalid video id' }, 400, cors);
  }

  const deleted = [];
  // List all files under highlights/{vid}/
  const listed = await env.BUCKET.list({ prefix: `highlights/${vid}/` });
  for (const obj of listed.objects) {
    await env.BUCKET.delete(obj.key);
    deleted.push(obj.key);
  }
  // Also delete thumbnails
  for (const thumbKey of [`highlights/thumbs/${vid}.jpg`, `thumbs/${vid}.jpg`]) {
    try { await env.BUCKET.delete(thumbKey); deleted.push(thumbKey); } catch {}
  }

  // Append to deletion log
  let log = [];
  try {
    const logObj = await env.BUCKET.get('highlights/deleted.json');
    if (logObj) log = await logObj.json();
  } catch {}
  log.push({
    video_id: vid,
    deleted_at: new Date().toISOString(),
    files_removed: deleted.length,
  });
  await env.BUCKET.put('highlights/deleted.json', JSON.stringify(log), {
    httpMetadata: { contentType: 'application/json' },
  });

  return jsonResponse({ ok: true, deleted: deleted.length, files: deleted }, 200, cors);
}

// ---------------------------------------------------------------------------
// Tags — per-session people tagging stored in highlights/tags.json
// ---------------------------------------------------------------------------

async function handleGetTags(env, cors) {
  try {
    const obj = await env.BUCKET.get('highlights/tags.json');
    if (obj) {
      return jsonResponse(await obj.json(), 200, cors);
    }
  } catch {}
  return jsonResponse({}, 200, cors);
}

async function handleSetTags(request, env, cors) {
  const body = await request.json();
  const { password, date, tags } = body;

  if (!password || password !== env.UPLOAD_PASSWORD) {
    return jsonResponse({ error: 'Invalid password' }, 403, cors);
  }

  if (!date || !Array.isArray(tags)) {
    return jsonResponse({ error: 'date and tags[] required' }, 400, cors);
  }

  // Read existing
  let allTags = {};
  try {
    const obj = await env.BUCKET.get('highlights/tags.json');
    if (obj) allTags = await obj.json();
  } catch {}

  // Update
  if (tags.length === 0) {
    delete allTags[date];
  } else {
    allTags[date] = tags;
  }

  await env.BUCKET.put('highlights/tags.json', JSON.stringify(allTags), {
    httpMetadata: { contentType: 'application/json' },
  });

  return jsonResponse({ ok: true, tags: allTags }, 200, cors);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateId() {
  const ts = Date.now().toString(36);
  const rand = Math.random().toString(36).substring(2, 8);
  return `${ts}_${rand}`;
}

function parseRange(header) {
  if (!header) return undefined;
  const match = header.match(/bytes=(\d+)-(\d*)/);
  if (!match) return undefined;
  const offset = parseInt(match[1]);
  const end = match[2] ? parseInt(match[2]) : undefined;
  if (end !== undefined) {
    return { offset, length: end - offset + 1 };
  }
  return { offset };
}

function jsonResponse(data, status = 200, headers = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}

// ---------------------------------------------------------------------------
// Auth — Sign in with Apple (PR 1)
//
// Flow:
//   1. iOS app obtains an Apple identity token via ASAuthorizationController.
//   2. iOS POSTs { identity_token } to /api/auth/apple.
//   3. We verify the token against Apple's JWKs (cached for 6h).
//   4. We compute user_hash = "u_" + first 8 hex of sha256(apple_sub).
//   5. We upsert users/<apple_sub>.json in R2.
//   6. We mint our own HS256 JWT (30-day TTL) and return it.
//   7. iOS uses that JWT as Bearer for all subsequent calls.
//
// Secrets / vars:
//   env.APPLE_BUNDLE_ID     — set in wrangler.toml [vars]
//   env.JWT_SIGNING_SECRET  — set via `wrangler secret put JWT_SIGNING_SECRET`
//
// TODO before App Store: gate /api/auth/apple behind an allowlist of
// approved Apple subs so anyone who installs the IPA can't sign in and
// upload. For now, any Apple ID can sign in.
// ---------------------------------------------------------------------------

const APPLE_JWKS_URL = 'https://appleid.apple.com/auth/keys';
const APPLE_ISSUER = 'https://appleid.apple.com';
const APPLE_JWKS_TTL_MS = 6 * 60 * 60 * 1000; // 6 hours
const OUR_JWT_TTL_SEC = 30 * 24 * 60 * 60;    // 30 days

let _appleJwksCache = null; // { keys, expiresAt }

async function fetchAppleJwks() {
  if (_appleJwksCache && _appleJwksCache.expiresAt > Date.now()) {
    return _appleJwksCache.keys;
  }
  const resp = await fetch(APPLE_JWKS_URL);
  if (!resp.ok) throw new Error(`Apple JWKS fetch failed: ${resp.status}`);
  const body = await resp.json();
  _appleJwksCache = {
    keys: body.keys,
    expiresAt: Date.now() + APPLE_JWKS_TTL_MS,
  };
  return body.keys;
}

function base64urlDecodeToBytes(s) {
  const b64 = s.replace(/-/g, '+').replace(/_/g, '/');
  const pad = b64.length % 4 === 0 ? '' : '='.repeat(4 - (b64.length % 4));
  const raw = atob(b64 + pad);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

function base64urlDecodeToString(s) {
  return new TextDecoder().decode(base64urlDecodeToBytes(s));
}

function base64urlEncodeBytes(bytes) {
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function base64urlEncodeString(s) {
  return base64urlEncodeBytes(new TextEncoder().encode(s));
}

async function verifyAppleIdentityToken(token, expectedBundleId) {
  const parts = token.split('.');
  if (parts.length !== 3) throw new Error('Malformed Apple token');
  const [headerB64, payloadB64, signatureB64] = parts;

  let header, payload;
  try {
    header = JSON.parse(base64urlDecodeToString(headerB64));
    payload = JSON.parse(base64urlDecodeToString(payloadB64));
  } catch {
    throw new Error('Malformed Apple token: bad base64/json');
  }

  if (header.alg !== 'RS256') {
    throw new Error(`Unsupported Apple token alg: ${header.alg}`);
  }

  const jwks = await fetchAppleJwks();
  const jwk = jwks.find((k) => k.kid === header.kid);
  if (!jwk) {
    // Possibly a key rotation we haven't seen — invalidate cache and retry once.
    _appleJwksCache = null;
    const retried = await fetchAppleJwks();
    const jwk2 = retried.find((k) => k.kid === header.kid);
    if (!jwk2) throw new Error(`Apple JWKS missing kid: ${header.kid}`);
    return verifyAppleIdentityTokenWithJwk(jwk2, headerB64, payloadB64, signatureB64, payload, expectedBundleId);
  }

  return verifyAppleIdentityTokenWithJwk(jwk, headerB64, payloadB64, signatureB64, payload, expectedBundleId);
}

async function verifyAppleIdentityTokenWithJwk(jwk, headerB64, payloadB64, signatureB64, payload, expectedBundleId) {
  const key = await crypto.subtle.importKey(
    'jwk',
    { kty: jwk.kty, n: jwk.n, e: jwk.e, alg: 'RS256', ext: true },
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
    false,
    ['verify'],
  );

  const signingInput = new TextEncoder().encode(`${headerB64}.${payloadB64}`);
  const signature = base64urlDecodeToBytes(signatureB64);
  const ok = await crypto.subtle.verify({ name: 'RSASSA-PKCS1-v1_5' }, key, signature, signingInput);
  if (!ok) throw new Error('Apple token signature invalid');

  if (payload.iss !== APPLE_ISSUER) {
    throw new Error(`Apple token iss mismatch: ${payload.iss}`);
  }
  if (payload.aud !== expectedBundleId) {
    throw new Error(`Apple token aud mismatch: ${payload.aud}`);
  }
  const nowSec = Math.floor(Date.now() / 1000);
  if (payload.exp && payload.exp < nowSec) throw new Error('Apple token expired');
  if (!payload.sub) throw new Error('Apple token missing sub');

  return payload;
}

async function userHashFromAppleSub(sub) {
  const data = new TextEncoder().encode(sub);
  const hash = await crypto.subtle.digest('SHA-256', data);
  const hex = [...new Uint8Array(hash)]
    .slice(0, 4)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  return `u_${hex}`;
}

async function hmacSha256Sign(secret, message) {
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const sig = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(message));
  return new Uint8Array(sig);
}

async function hmacSha256Verify(secret, message, sigBytes) {
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['verify'],
  );
  return crypto.subtle.verify('HMAC', key, sigBytes, new TextEncoder().encode(message));
}

async function signOurJWT(claims, secret) {
  const header = { alg: 'HS256', typ: 'JWT' };
  const h = base64urlEncodeString(JSON.stringify(header));
  const p = base64urlEncodeString(JSON.stringify(claims));
  const sig = await hmacSha256Sign(secret, `${h}.${p}`);
  const s = base64urlEncodeBytes(sig);
  return `${h}.${p}.${s}`;
}

async function verifyOurJWT(jwt, secret) {
  if (!jwt) throw new Error('Missing JWT');
  const parts = jwt.split('.');
  if (parts.length !== 3) throw new Error('Malformed JWT');
  const [h, p, s] = parts;
  const sigBytes = base64urlDecodeToBytes(s);
  const ok = await hmacSha256Verify(secret, `${h}.${p}`, sigBytes);
  if (!ok) throw new Error('JWT signature invalid');
  const claims = JSON.parse(base64urlDecodeToString(p));
  const nowSec = Math.floor(Date.now() / 1000);
  if (claims.exp && claims.exp < nowSec) throw new Error('JWT expired');
  return claims;
}

// Authenticate an iPhone upload — accepts EITHER the legacy
// IPHONE_UPLOAD_TOKEN (Mac uploader) OR a user JWT (iOS app).
// Returns:
//   { ok: true, kind: 'shared' }
//   { ok: true, kind: 'user', user_hash }
//   { ok: false }
async function authenticateIphoneUpload(request, env) {
  const auth = request.headers.get('authorization') || '';
  if (!auth.startsWith('Bearer ')) return { ok: false };
  const token = auth.slice(7).trim();

  // Shared token (Mac uploader, single string)
  if (env.IPHONE_UPLOAD_TOKEN && token === env.IPHONE_UPLOAD_TOKEN) {
    return { ok: true, kind: 'shared' };
  }

  // User JWT (iOS app, three dot-separated parts)
  if (env.JWT_SIGNING_SECRET && token.split('.').length === 3) {
    try {
      const claims = await verifyOurJWT(token, env.JWT_SIGNING_SECRET);
      return { ok: true, kind: 'user', user_hash: claims.sub };
    } catch {
      return { ok: false };
    }
  }
  return { ok: false };
}

// Extract authenticated user from the Authorization header.
// Returns { kind: 'user', user_hash, claims } or { kind: null }.
// Future PRs (chunked upload routes) will use this to attribute uploads.
async function authenticateUser(request, env) {
  const auth = request.headers.get('authorization') || '';
  if (!auth.startsWith('Bearer ')) return { kind: null };
  const token = auth.slice(7).trim();
  // Heuristic: our JWTs have three dot-separated parts; the legacy
  // IPHONE_UPLOAD_TOKEN is opaque — don't waste an HMAC verify on it.
  if (token.split('.').length !== 3) return { kind: null };
  if (!env.JWT_SIGNING_SECRET) return { kind: null };
  try {
    const claims = await verifyOurJWT(token, env.JWT_SIGNING_SECRET);
    return { kind: 'user', user_hash: claims.sub, claims };
  } catch {
    return { kind: null };
  }
}

// POST /api/auth/apple
// Body:   { identity_token: string, nonce?: string }
// Returns 200: { jwt, user_hash, gallery_url, expires_at }
async function handleAuthApple(request, env, cors) {
  if (!env.JWT_SIGNING_SECRET) {
    return jsonResponse({ error: 'Server not configured (missing JWT secret)' }, 500, cors);
  }
  if (!env.APPLE_BUNDLE_ID) {
    return jsonResponse({ error: 'Server not configured (missing bundle id)' }, 500, cors);
  }

  const body = await request.json().catch(() => ({}));
  const idToken = body.identity_token;
  if (!idToken || typeof idToken !== 'string') {
    return jsonResponse({ error: 'identity_token required' }, 400, cors);
  }

  let applePayload;
  try {
    applePayload = await verifyAppleIdentityToken(idToken, env.APPLE_BUNDLE_ID);
  } catch (e) {
    return jsonResponse(
      { error: 'Apple token verification failed', detail: e.message },
      401, cors,
    );
  }

  const appleSub = applePayload.sub;
  const userHash = await userHashFromAppleSub(appleSub);

  // Allowlist check.
  //   - If users/_allowlist.json is absent → app is open (default).
  //   - If present and { open: true } → still open.
  //   - If present and { open: false } → only subs in `subs` array
  //     (or already-existing users) can sign in.
  //   - Before the App Store wider release, flip `open: false` and
  //     add invited Apple subs to keep randos out.
  // We need to peek at the existing user record to know if they're
  // banned (admin-tombstoned) vs self-deleted. Pull it once, use it
  // for both allowlist and rejoin logic below.
  const userKey = `users/${appleSub}.json`;
  const existingObj = await env.BUCKET.get(userKey);
  const existingRecord = existingObj ? await existingObj.json() : null;
  const isBanned = !!(existingRecord && existingRecord.status === 'banned');

  try {
    const allowlistObj = await env.BUCKET.get('users/_allowlist.json');
    if (allowlistObj) {
      const allowlist = await allowlistObj.json();
      if (allowlist.open === false) {
        const subs = Array.isArray(allowlist.subs) ? allowlist.subs : [];
        const isAllowedSub = subs.includes(appleSub);
        // Admin-banned users (status:'banned') NEVER pass the allowlist,
        // even though their user record still exists. They must be
        // explicitly re-added to subs[] AND have their status reset.
        if (isBanned) {
          return jsonResponse(
            { error: 'Not approved', detail: 'Your account has been banned by the administrator.' },
            403, cors,
          );
        }
        if (!isAllowedSub && !existingRecord) {
          return jsonResponse(
            { error: 'Not approved', detail: 'Your account is not on the invite list. Ask the administrator to add you.' },
            403, cors,
          );
        }
      }
    }
  } catch {
    // Allowlist read failed — fail open (don't lock out legitimate users
    // because of an R2 hiccup). Log if you wire structured logging later.
  }

  const nowIso = new Date().toISOString();
  let userRecord;
  if (existingRecord) {
    userRecord = existingRecord;
    userRecord.last_seen = nowIso;
    userRecord.user_hash = userHash; // keep in sync if the hash impl changes
    // If they previously SELF-deleted (status === 'deleted'), allow rejoin.
    // status === 'banned' is admin-only and caught above.
    if (userRecord.deleted_at && userRecord.status !== 'banned') {
      delete userRecord.deleted_at;
      userRecord.status = 'active';
      userRecord.rejoined_at = nowIso;
    }
  } else {
    userRecord = {
      apple_sub: appleSub,
      user_hash: userHash,
      created_at: nowIso,
      last_seen: nowIso,
      video_count: 0,
      status: 'active',
      // email/name are only sent by Apple on first sign-in — keep what we get.
      email: applePayload.email || null,
      email_verified:
        applePayload.email_verified === 'true' || applePayload.email_verified === true || null,
    };
  }
  await env.BUCKET.put(userKey, JSON.stringify(userRecord), {
    httpMetadata: { contentType: 'application/json' },
  });

  const nowSec = Math.floor(Date.now() / 1000);
  const claims = {
    sub: userHash,
    apple_sub: appleSub,
    iat: nowSec,
    exp: nowSec + OUR_JWT_TTL_SEC,
    scope: 'upload',
  };
  const jwt = await signOurJWT(claims, env.JWT_SIGNING_SECRET);

  return jsonResponse(
    {
      jwt,
      user_hash: userHash,
      gallery_url: `https://tennis.playfullife.com/u/${userHash}`,
      expires_at: claims.exp,
    },
    200, cors,
  );
}

// DELETE /api/account — Apple App Review requires in-app account deletion.
// Tombstones the users/<sub>.json record and deletes ALL videos that
// were uploaded with that user_hash. Gallery regen happens on the next
// pipeline run (deleted videos drop out then); cached index.html may
// briefly still show them.
async function handleDeleteAccount(request, env, cors) {
  const authResult = await authenticateUser(request, env);
  if (authResult.kind !== 'user') {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }

  const appleSub = authResult.claims.apple_sub;
  const userHash = authResult.claims.sub;
  const userKey = `users/${appleSub}.json`;
  const userObj = await env.BUCKET.get(userKey);
  if (!userObj) {
    return jsonResponse({ error: 'User not found' }, 404, cors);
  }
  const user = await userObj.json();

  // Find every marker tagged with this user_hash and delete its file family.
  const deletedVideos = [];
  let cursor;
  do {
    const listed = await env.BUCKET.list({ prefix: 'uploads/', cursor });
    for (const obj of listed.objects) {
      if (!obj.key.endsWith('.json') || obj.key.includes('_inflight_')) continue;
      try {
        const m = await env.BUCKET.get(obj.key);
        if (!m) continue;
        const marker = await m.json();
        if (marker.uploaded_by !== userHash) continue;
        const vid = marker.video_id;
        // Source file (either .mov or .mp4)
        for (const ext of ['mov', 'mp4']) {
          await env.BUCKET.delete(`source/${vid}.${ext}`).catch(() => {});
        }
        // Processed outputs
        let pCursor;
        do {
          const pList = await env.BUCKET.list({ prefix: `processed/${vid}/`, cursor: pCursor });
          for (const p of pList.objects) {
            await env.BUCKET.delete(p.key).catch(() => {});
          }
          pCursor = pList.truncated ? pList.cursor : undefined;
        } while (pCursor);
        // Thumbnail
        await env.BUCKET.delete(`highlights/thumbs/${vid}.jpg`).catch(() => {});
        // Marker itself
        await env.BUCKET.delete(obj.key).catch(() => {});
        deletedVideos.push(vid);
      } catch {}
    }
    cursor = listed.truncated ? listed.cursor : undefined;
  } while (cursor);

  // Tombstone the user record so audit history remains. Sign-in re-uses
  // the same record (deleted_at cleared) if they come back later.
  user.deleted_at = new Date().toISOString();
  user.status = 'deleted';
  await env.BUCKET.put(userKey, JSON.stringify(user), {
    httpMetadata: { contentType: 'application/json' },
  });

  return jsonResponse(
    { deleted: true, videos_removed: deletedVideos.length, video_ids: deletedVideos },
    200, cors,
  );
}

// GET /api/me — current user's profile (Bearer JWT required).
async function handleMe(request, env, cors) {
  const authResult = await authenticateUser(request, env);
  if (authResult.kind !== 'user') {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }

  const appleSub = authResult.claims.apple_sub;
  const userObj = await env.BUCKET.get(`users/${appleSub}.json`);
  if (!userObj) {
    return jsonResponse({ error: 'User not found' }, 404, cors);
  }
  const user = await userObj.json();
  // Treat tombstoned users as "not found" so the app routes back to sign-in.
  if (user.deleted_at) {
    return jsonResponse({ error: 'Account deleted' }, 401, cors);
  }
  return jsonResponse(
    {
      user_hash: user.user_hash,
      video_count: user.video_count || 0,
      gallery_url: `https://tennis.playfullife.com/u/${user.user_hash}`,
      created_at: user.created_at,
    },
    200, cors,
  );
}
