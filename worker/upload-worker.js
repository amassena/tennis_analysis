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

    // PR-F: public share-link viewer at /v/<token>. No auth — the token
    // IS the bearer. Serves a tiny HTML player page that embeds the
    // owner's timeline.mp4 (resolved via the share record).
    const shareViewMatch = path.match(/^\/v\/([A-Za-z0-9_-]+)$/);
    if (shareViewMatch && (request.method === 'GET' || request.method === 'HEAD')) {
      return handleViewShare(request, env, shareViewMatch[1]);
    }
    // Streamed source for the share-page <video src=…> tag, served
    // without the per-user auth gate (because the token already
    // authenticated the watcher). Path: /v/<token>/video
    const shareMediaMatch = path.match(/^\/v\/([A-Za-z0-9_-]+)\/video$/);
    if (shareMediaMatch && (request.method === 'GET' || request.method === 'HEAD')) {
      return handleShareMedia(request, env, shareMediaMatch[1]);
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
  let key;
  if (path === '/' || path === '/index.html') {
    key = 'static/root-landing.html';
  } else if (path === '/admin' || path === '/admin/' || path === '/admin.html') {
    // Accept the trailing-slash variant too — in-app browsers (e.g. the Claude
    // app's web view) append "/", which otherwise 404'd.
    key = 'static/admin.html';
  } else if (path === '/privacy' || path === '/privacy.html') {
    key = 'static/privacy.html';
  } else if (path === '/support' || path === '/support.html') {
    key = 'static/support.html';
  } else {
    key = path.slice(1);
  }

  // Per-user JWT fallback: if the request carries a valid `tennis_jwt`
  // cookie/bearer, treat any non-static path as a request for the
  // user's per-user-prefixed copy of that asset. Lets the legacy gallery
  // URLs (/thumbs/<vid>.jpg, /<vid>/timeline.mp4) keep working even
  // when CDN/WKWebView quirks drop the per-user prefix in subresource
  // fetches. Matches the user's "worker-side JWT lookup" option.
  let perUserKey = null;
  if (!key.startsWith('static/') && !key.startsWith('uploads/') &&
      !key.startsWith('users/')) {
    const cookieToken = readCookie(request, 'tennis_jwt');
    const headerAuth = (request.headers.get('authorization') || '').startsWith('Bearer ')
      ? request.headers.get('authorization').slice(7).trim() : null;
    const token = cookieToken || headerAuth;
    if (token && env.JWT_SIGNING_SECRET) {
      try {
        const claims = await verifyOurJWT(token, env.JWT_SIGNING_SECRET);
        const sub = claims.sub;
        if (sub && sub.startsWith('u_') && sub.length === 10) {
          // Strip a leading "highlights/" so we don't end up with
          // highlights/<sub>/highlights/<key>.
          const rel = key.startsWith('highlights/') ? key.slice('highlights/'.length) : key;
          perUserKey = `highlights/${sub}/${rel}`;
        }
      } catch {}
    }
  }

  if (request.method === 'HEAD') {
    return handleHead(env, key, perUserKey);
  }

  return serveR2Object(request, env, key, {
    fallbackHighlightsPrefix: true,
    fallbackKey: perUserKey,
    // Per-user fallback responses are owner-scoped; opt out of edge cache.
    private: perUserKey != null,
  });
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

  // Per-user fallback (auth-resolved). Used when neither the literal
  // path nor highlights/<key> exist on the public layout — the asset
  // probably lives under highlights/<sub>/...
  if (!obj && opts.fallbackKey) {
    try {
      obj = await env.BUCKET.get(opts.fallbackKey, getOpts);
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
  // Auth-gated assets: cache only in the browser, never at the edge.
  // CF's default cache key does NOT include cookies, so a public 401
  // for /u/<hash>/thumbs/<vid>.jpg can be served back to a signed-in
  // user if we let the edge cache. `private` opts CF out.
  const cacheScope = opts.private ? 'private' : 'public';
  if (ct.startsWith('video/')) {
    headers.set('cache-control', `${cacheScope}, max-age=86400`);
  } else if (isHtml) {
    headers.set('cache-control', 'no-store, no-cache, must-revalidate, max-age=0');
    headers.set('cdn-cache-control', 'no-store');
  } else {
    headers.set('cache-control', `${cacheScope}, max-age=3600`);
  }
  if (opts.private) {
    headers.set('cdn-cache-control', 'no-store');
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
      {
        status: 401,
        headers: {
          'content-type': 'text/plain; charset=utf-8',
          // Critical: never let CF/edge cache the 401. Without this,
          // browsers that fetch /u/<hash>/thumbs/<vid>.jpg before the
          // auth cookie is set can get a cached 401 even after sign-in.
          'cache-control': 'no-store, no-cache, must-revalidate, max-age=0',
          'cdn-cache-control': 'no-store',
        },
      },
    );
  }
  if (claims.sub !== userHash && !isAdminUser(env, claims.sub)) {
    return new Response('Forbidden', {
      status: 403,
      headers: {
        'cache-control': 'no-store, no-cache, must-revalidate, max-age=0',
        'cdn-cache-control': 'no-store',
      },
    });
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
  return serveR2Object(request, env, key, { private: true });
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

async function handleHead(env, key, perUserKey) {
  let obj = await env.BUCKET.head(key);
  if (!obj && !key.startsWith('highlights/') && !key.startsWith('uploads/')) {
    obj = await env.BUCKET.head('highlights/' + key);
  }
  if (!obj && perUserKey) {
    obj = await env.BUCKET.head(perUserKey);
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

    // Magic-link auth: request sends a one-time URL to the user's email;
    // consume verifies the token, mints a JWT, sets the auth cookie, and
    // redirects to the user's gallery.
    if (path === '/api/auth/magic/request' && request.method === 'POST') {
      return await handleMagicRequest(request, env, cors);
    }
    if (path === '/api/auth/magic/consume' && request.method === 'GET') {
      return await handleMagicConsume(request, env, cors);
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

    if (path === '/api/admin/queue' && request.method === 'GET') {
      return await handleAdminQueue(request, env, cors);
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

    // PR-F: per-video share link. POST mints a token for the caller's
    // video; GET /v/<token> serves a public player page.
    const shareMatch = path.match(/^\/api\/video\/([^/]+)\/share$/);
    if (shareMatch && request.method === 'POST') {
      return await handleCreateShare(request, env, cors, shareMatch[1]);
    }

    // PR-D: rename a video. Owner JWT (or admin) only.
    const renameMatch = path.match(/^\/api\/video\/([^/]+)\/rename$/);
    if (renameMatch && request.method === 'POST') {
      return await handleRenameVideo(request, env, cors, renameMatch[1]);
    }

    // GET /api/u/<hash>/recent — user's recent upload markers (PR-B).
    const recentMatch = path.match(/^\/api\/u\/(u_[a-f0-9]{8})\/recent$/);
    if (recentMatch && request.method === 'GET') {
      return await handleUserRecent(request, env, cors, recentMatch[1]);
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

// Recover the set of already-uploaded parts for an in-flight upload from the
// discrete uploads/_parts_{vid}/ objects written by handleIphonePart. Reads
// part metadata from customMetadata (no GET per part) and returns parts sorted
// by partNumber. Server-authoritative: works even if the client lost its
// local state (reinstall) — re-picking the same asset resumes from here.
async function listUploadedParts(env, videoId) {
  const prefix = `uploads/_parts_${videoId}/`;
  const out = [];
  let cursor;
  do {
    const res = await env.BUCKET.list({ prefix, include: ['customMetadata'], cursor });
    for (const obj of res.objects || []) {
      const m = obj.customMetadata || {};
      const pn = Number(m.pn);
      if (!pn) continue;
      out.push({ partNumber: pn, etag: m.etag, size: Number(m.size) || 0 });
    }
    cursor = res.truncated ? res.cursor : undefined;
  } while (cursor);
  out.sort((a, b) => a.partNumber - b.partNumber);
  return out;
}

async function handleIphoneInit(request, env, cors) {
  const auth = await authenticateIphoneUpload(request, env);
  if (!auth.ok) {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }
  const body = await request.json().catch(() => ({}));
  const { asset_id: assetId, filename } = body;
  const createdAt = body.created_at || '';
  const totalBytes = Number(body.total_bytes) || 0;
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

  // Resume: if an in-flight multipart already exists for this video, return it
  // (plus the parts already uploaded) instead of starting a NEW multipart —
  // which would orphan the prior progress (the 84% IMG_1291 loss). This makes
  // re-picking the same asset after a stall/kill/reinstall continue from where
  // it left off, with no client-side state required.
  const inflightKey = `uploads/_inflight_${videoId}.json`;
  try {
    const existingState = await env.BUCKET.get(inflightKey);
    if (existingState) {
      const st = await existingState.json();
      if (st.r2_upload_id) {
        const parts = await listUploadedParts(env, videoId);
        // Backfill total_bytes for markers created before we tracked it, so
        // complete-time byte validation applies to resumed uploads too.
        if (!st.total_bytes && totalBytes > 0) {
          st.total_bytes = totalBytes;
          try {
            await env.BUCKET.put(inflightKey, JSON.stringify(st),
              { httpMetadata: { contentType: 'application/json' } });
          } catch {}
        }
        return jsonResponse(
          {
            video_id: videoId,
            upload_id: videoId,
            r2_key: st.r2_key,
            asset_id: assetId,
            status: 'resume',
            parts,
            total_bytes: st.total_bytes || totalBytes,
          },
          200, cors,
        );
      }
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
      total_bytes: totalBytes,
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

  // Record this part as a discrete R2 object so init can resume and complete
  // can validate the whole file — server-authoritative, independent of any
  // client-side state (which is lost on reinstall). Discrete keys (one per
  // partNumber) are race-free: the 3 concurrent part uploads each write their
  // own key, so there's no read-modify-write clobber on the shared marker.
  const size = Number(request.headers.get('content-length')) || 0;
  const partKey = `uploads/_parts_${uploadId}/${String(partNumber).padStart(5, '0')}.json`;
  try {
    await env.BUCKET.put(
      partKey,
      JSON.stringify({ partNumber: part.partNumber, etag: part.etag, size }),
      {
        httpMetadata: { contentType: 'application/json' },
        // Mirror into customMetadata so a list({include:['customMetadata']})
        // recovers the part set without a GET per part.
        customMetadata: { pn: String(part.partNumber), etag: part.etag, size: String(size) },
      },
    );
  } catch {}

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

  const body = await request.json().catch(() => ({}));
  const clientParts = Array.isArray(body.parts) ? body.parts : [];

  // Server-authoritative completion. Prefer the parts we recorded as they
  // landed (uploads/_parts_{vid}/) over the client's list, and validate the
  // whole file is present BEFORE finalizing — a short part set would otherwise
  // assemble a truncated, corrupt video and report success (the latent bug
  // behind the IMG_1291 incident). Reject incomplete with 409 so the client
  // keeps/resumes uploading instead of believing it's done.
  const serverParts = await listUploadedParts(env, uploadId);
  let finalParts;
  if (serverParts.length > 0) {
    const contiguous = serverParts.every((p, i) => p.partNumber === i + 1);
    if (!contiguous) {
      return jsonResponse(
        { error: 'Incomplete upload', detail: 'non-contiguous parts',
          have: serverParts.map((p) => p.partNumber), status: 'incomplete' },
        409, cors,
      );
    }
    const expected = Number(state.total_bytes) || 0;
    const sum = serverParts.reduce((a, p) => a + (p.size || 0), 0);
    if (expected > 0 && sum !== expected) {
      return jsonResponse(
        { error: 'Incomplete upload',
          detail: `have ${sum} of ${expected} bytes (${serverParts.length} parts)`,
          bytes_have: sum, bytes_expected: expected, parts_have: serverParts.length,
          status: 'incomplete' },
        409, cors,
      );
    }
    finalParts = serverParts.map((p) => ({ partNumber: p.partNumber, etag: p.etag }));
  } else if (clientParts.length > 0) {
    // Backward-compat: uploads started before per-part tracking existed.
    finalParts = clientParts.map((p) => ({ partNumber: p.partNumber, etag: p.etag }));
  } else {
    return jsonResponse({ error: 'parts[] required' }, 400, cors);
  }

  const upload = env.BUCKET.resumeMultipartUpload(state.r2_key, state.r2_upload_id);
  await upload.complete(finalParts);

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

  // Clean up in-flight state + the per-part records.
  try { await env.BUCKET.delete(stateKey); } catch {}
  try {
    const prefix = `uploads/_parts_${uploadId}/`;
    let cursor;
    do {
      const res = await env.BUCKET.list({ prefix, cursor });
      const keys = (res.objects || []).map((o) => o.key);
      if (keys.length) await env.BUCKET.delete(keys);
      cursor = res.truncated ? res.cursor : undefined;
    } while (cursor);
  } catch {}

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
  // List top-level upload metadata JSONs (delimiter:'/' already excludes the
  // _parts_<vid>/ subdirs). Skip in-flight init state files (ghost "Queued"
  // rows) and the allowlist config.
  const listed = await env.BUCKET.list({ prefix: 'uploads/', delimiter: '/' });
  const keys = listed.objects
    .filter((obj) => obj.key.endsWith('.json')
      && !obj.key.includes('_inflight_')
      && obj.key !== 'uploads/_allowlist.json')
    .map((obj) => obj.key);

  // Parallel fetch — was a sequential GET per marker, which slowed linearly as
  // the catalog grew.
  const metas = await Promise.all(keys.map(async (k) => {
    try { const o = await env.BUCKET.get(k); return o ? await o.json() : null; }
    catch { return null; }
  }));

  const items = [];
  for (const meta of metas) {
    if (!meta) continue;
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

  // Sort by upload time, newest first
  items.sort((a, b) => (b.uploaded_at || '').localeCompare(a.uploaded_at || ''));

  return jsonResponse({ queue: items }, 200, cors);
}

// GET /api/admin/queue — ops dashboard data. Admin-only. Unlike /api/queue
// this INCLUDES in-flight uploads (_inflight_ markers) and per-user
// attribution (user_hash), so admins can see what's uploading right now,
// by whom, and where the pipeline is stuck.
async function handleAdminQueue(request, env, cors) {
  const noCache = { ...cors, 'cache-control': 'no-store', 'cdn-cache-control': 'no-store' };
  const cookieToken = readCookie(request, 'tennis_jwt');
  const headerAuth = (request.headers.get('authorization') || '').startsWith('Bearer ')
    ? request.headers.get('authorization').slice(7).trim() : null;
  const token = cookieToken || headerAuth;
  let claims = null;
  if (token && env.JWT_SIGNING_SECRET) {
    try { claims = await verifyOurJWT(token, env.JWT_SIGNING_SECRET); } catch {}
  }
  if (!claims) return jsonResponse({ error: 'Unauthorized' }, 401, noCache);
  if (!isAdminUser(env, claims.sub)) return jsonResponse({ error: 'Forbidden' }, 403, noCache);

  // Collect candidate keys (cheap list), then GET in PARALLEL. Exclude
  // _parts_ records (per-part upload tracking) — they live under uploads/ and
  // would otherwise show as junk "00001" rows and add a GET per chunk.
  const entries = [];
  let cursor;
  do {
    const page = await env.BUCKET.list({ prefix: 'uploads/', cursor });
    for (const obj of page.objects) {
      if (!obj.key.endsWith('.json')) continue;
      if (obj.key.includes('_parts_')) continue;
      if (obj.key === 'uploads/_allowlist.json') continue;
      entries.push({ key: obj.key, inflight: obj.key.includes('_inflight_') });
    }
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);

  const items = (await Promise.all(entries.map(async ({ key, inflight }) => {
    try {
      const m = await (await env.BUCKET.get(key)).json();
      return {
        video_id: m.id || key.split('/').pop().replace('_inflight_', '').replace('.json', ''),
        filename: m.filename || m.url || 'Unknown',
        user_hash: m.user_hash || m.uploaded_by || null,
        status: inflight ? 'uploading' : (m.status || 'unknown'),
        stage: m.stage || null,
        progress: m.progress ?? null,
        uploaded_at: m.uploaded_at || m.created_at || null,
        updated_at: m.updated_at || m.completed_at || m.uploaded_at || null,
        error: m.error || null,
        inflight,
      };
    } catch { return null; }
  }))).filter(Boolean);

  items.sort((a, b) => (b.uploaded_at || '').localeCompare(a.uploaded_at || ''));

  // Per-user rollup
  const byUser = {};
  for (const it of items) {
    const u = it.user_hash || 'unknown';
    byUser[u] = byUser[u] || { user_hash: u, total: 0, uploading: 0, processing: 0, failed: 0, complete: 0 };
    byUser[u].total++;
    if (it.inflight) byUser[u].uploading++;
    else if (it.status === 'failed') byUser[u].failed++;
    else if (it.status === 'complete') byUser[u].complete++;
    else byUser[u].processing++;
  }

  return jsonResponse({ items, users: Object.values(byUser), generated_at: new Date().toISOString() }, 200, noCache);
}

// ---------------------------------------------------------------------------
// Delete a video and all its files (videos, thumbnail, meta.json)
// ---------------------------------------------------------------------------

// GET /api/u/<hash>/recent — return the user's recent upload markers
// (default last 25), sorted by uploaded_at desc. Auth: JWT cookie or
// Bearer whose sub matches the URL hash, or admin.
async function handleUserRecent(request, env, cors, userHash) {
  const cookieToken = readCookie(request, 'tennis_jwt');
  const headerAuth = (request.headers.get('authorization') || '').startsWith('Bearer ')
    ? request.headers.get('authorization').slice(7).trim() : null;
  const token = cookieToken || headerAuth;
  let claims = null;
  if (token && env.JWT_SIGNING_SECRET) {
    try { claims = await verifyOurJWT(token, env.JWT_SIGNING_SECRET); } catch {}
  }
  if (!claims) {
    return jsonResponse({ error: 'Unauthorized' }, 401,
      { ...cors, 'cache-control': 'no-store', 'cdn-cache-control': 'no-store' });
  }
  if (claims.sub !== userHash && !isAdminUser(env, claims.sub)) {
    return jsonResponse({ error: 'Forbidden' }, 403,
      { ...cors, 'cache-control': 'no-store', 'cdn-cache-control': 'no-store' });
  }

  const url = new URL(request.url);
  const limit = Math.max(1, Math.min(50, parseInt(url.searchParams.get('limit') || '25')));

  // Collect candidate marker keys (cheap list), then GET them in PARALLEL.
  // Sequential GETs made this 2-3s for ~40 markers, which on-device read as a
  // hang. Also exclude _parts_ records (per-part upload tracking from the
  // resumable-upload work) — they live under uploads/ and would otherwise add
  // a GET per 50MB chunk (~140 junk GETs for a 7GB upload).
  const keys = [];
  let cursor;
  do {
    const page = await env.BUCKET.list({ prefix: 'uploads/', cursor });
    for (const obj of page.objects) {
      if (!obj.key.endsWith('.json')) continue;
      if (obj.key.includes('_inflight_')) continue;
      if (obj.key.includes('_parts_')) continue;
      if (obj.key === 'uploads/_allowlist.json') continue;
      keys.push(obj.key);
    }
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);

  const markers = await Promise.all(keys.map(async (k) => {
    try { const m = await env.BUCKET.get(k); return m ? await m.json() : null; }
    catch { return null; }
  }));

  const items = [];
  for (const marker of markers) {
    if (!marker) continue;
    const owner = marker.user_hash || marker.uploaded_by;
    if (owner !== userHash) continue;
    items.push({
      video_id: marker.video_id || marker.id,
      filename: marker.filename || '',
      status: marker.status || 'queued',
      stage: marker.stage || null,
      progress: marker.progress != null ? marker.progress : null,
      uploaded_at: marker.uploaded_at || marker.created_at || null,
      updated_at: marker.updated_at || marker.completed_at || marker.uploaded_at || null,
      error: marker.error || null,
      video_url: marker.video_url || null,
      // Browser-resolvable URL once processing is done; the app can
      // tap-through directly into the gallery WebView.
      gallery_url: marker.status === 'complete'
        ? `https://tennis.playfullife.com/u/${userHash}#${marker.video_id || marker.id}`
        : null,
    });
  }

  items.sort((a, b) => {
    const ta = a.uploaded_at ? Date.parse(a.uploaded_at) : 0;
    const tb = b.uploaded_at ? Date.parse(b.uploaded_at) : 0;
    return tb - ta;
  });

  return jsonResponse(
    { user_hash: userHash, count: items.length, items: items.slice(0, limit) },
    200,
    { ...cors, 'cache-control': 'no-store', 'cdn-cache-control': 'no-store' },
  );
}

// ---------------------------------------------------------------------------
// PR-D — rename a video (display_name in meta.json)
// ---------------------------------------------------------------------------
async function handleRenameVideo(request, env, cors, vid) {
  if (!/^[A-Za-z0-9_-]+$/.test(vid)) {
    return jsonResponse({ error: 'Invalid video id' }, 400, cors);
  }
  const body = await request.json().catch(() => ({}));
  const raw = (body.display_name || '').toString().trim();
  // Empty string clears the display name back to the default (vid).
  // Cap length to keep gallery cards readable.
  const displayName = raw.slice(0, 80);

  // Auth: owner JWT or admin.
  const cookieToken = readCookie(request, 'tennis_jwt');
  const headerAuth = (request.headers.get('authorization') || '').startsWith('Bearer ')
    ? request.headers.get('authorization').slice(7).trim() : null;
  const token = cookieToken || headerAuth;
  let claims = null;
  if (token && env.JWT_SIGNING_SECRET) {
    try { claims = await verifyOurJWT(token, env.JWT_SIGNING_SECRET); } catch {}
  }
  if (!claims) {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }

  // Resolve owner from marker.
  let owner = null;
  try {
    const markerObj = await env.BUCKET.get(`uploads/${vid}.json`);
    if (markerObj) {
      const marker = await markerObj.json();
      owner = marker.user_hash || marker.uploaded_by || null;
    }
  } catch {}
  if (!owner) return jsonResponse({ error: 'Video not found' }, 404, cors);
  const isOwner = claims.sub === owner;
  const isAdmin = isAdminUser(env, claims.sub);
  if (!isOwner && !isAdmin) {
    return jsonResponse({ error: 'Only the owner can rename this video' }, 403, cors);
  }

  // Patch meta.json. Falls back to creating one if missing.
  const metaKey = `highlights/${owner}/${vid}/meta.json`;
  let meta = {};
  try {
    const m = await env.BUCKET.get(metaKey);
    if (m) meta = await m.json();
  } catch {}
  if (displayName) meta.display_name = displayName;
  else delete meta.display_name;
  meta.display_name_updated_at = new Date().toISOString();
  await env.BUCKET.put(metaKey, JSON.stringify(meta), {
    httpMetadata: { contentType: 'application/json' },
  });

  return jsonResponse(
    { ok: true, vid, display_name: meta.display_name || null },
    200, cors,
  );
}

// ---------------------------------------------------------------------------
// PR-F — per-video share links
// ---------------------------------------------------------------------------
//
// shares/<token>.json:
//   { token, vid, owner, created_at, click_count, last_seen_at? }
//
// Flow:
//   1. Owner taps Share in the gallery → POST /api/video/<vid>/share
//   2. Worker mints a 16-byte token, stores the record, returns the URL.
//   3. Anyone with the URL hits GET /v/<token> → tiny HTML player page.
//   4. The page's <video> tag fetches GET /v/<token>/video which streams
//      the timeline.mp4 from highlights/<owner>/<vid>/<vid>_timeline.mp4
//      without requiring the watcher to be signed in.
//
// Tokens don't expire — the owner can revoke by listing/deleting the
// `shares/<token>.json` record (TODO: revoke UI). They're 22 chars of
// base64url so brute force is infeasible.
const SHARE_TOKEN_BYTES = 16;

async function handleCreateShare(request, env, cors, vid) {
  if (!/^[A-Za-z0-9_-]+$/.test(vid)) {
    return jsonResponse({ error: 'Invalid video id' }, 400, cors);
  }
  // Auth: owner JWT or admin JWT.
  const cookieToken = readCookie(request, 'tennis_jwt');
  const headerAuth = (request.headers.get('authorization') || '').startsWith('Bearer ')
    ? request.headers.get('authorization').slice(7).trim() : null;
  const token = cookieToken || headerAuth;
  let claims = null;
  if (token && env.JWT_SIGNING_SECRET) {
    try { claims = await verifyOurJWT(token, env.JWT_SIGNING_SECRET); } catch {}
  }
  if (!claims) {
    return jsonResponse({ error: 'Unauthorized' }, 401, cors);
  }

  // Marker → owner.
  let owner = null;
  try {
    const markerObj = await env.BUCKET.get(`uploads/${vid}.json`);
    if (markerObj) {
      const marker = await markerObj.json();
      owner = marker.user_hash || marker.uploaded_by || null;
    }
  } catch {}
  if (!owner) {
    return jsonResponse({ error: 'Video not found' }, 404, cors);
  }
  const isOwner = claims.sub === owner;
  const isAdmin = isAdminUser(env, claims.sub);
  if (!isOwner && !isAdmin) {
    return jsonResponse({ error: 'Only the video owner can share it' }, 403, cors);
  }

  // Existing share? Reuse so repeated taps don't generate new tokens.
  const existing = await findExistingShareForVid(env, vid);
  if (existing) {
    return jsonResponse(
      { url: `https://tennis.playfullife.com/v/${existing.token}`,
        token: existing.token, vid, owner, reused: true },
      200, cors,
    );
  }

  const tok = base64urlEncodeBytes(crypto.getRandomValues(new Uint8Array(SHARE_TOKEN_BYTES)));
  const record = {
    token: tok,
    vid,
    owner,
    created_by: claims.sub,
    created_at: new Date().toISOString(),
    click_count: 0,
  };
  await env.BUCKET.put(`shares/${tok}.json`, JSON.stringify(record), {
    httpMetadata: { contentType: 'application/json' },
  });
  return jsonResponse(
    { url: `https://tennis.playfullife.com/v/${tok}`, token: tok, vid, owner },
    200, cors,
  );
}

// Linear scan — there shouldn't be many shares. Could index later.
async function findExistingShareForVid(env, vid) {
  let cursor;
  do {
    const page = await env.BUCKET.list({ prefix: 'shares/', cursor });
    for (const obj of page.objects) {
      try {
        const r = await env.BUCKET.get(obj.key);
        if (!r) continue;
        const rec = await r.json();
        if (rec.vid === vid) return rec;
      } catch {}
    }
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  return null;
}

async function loadShareRecord(env, tok) {
  if (!tok || !/^[A-Za-z0-9_-]+$/.test(tok)) return null;
  try {
    const obj = await env.BUCKET.get(`shares/${tok}.json`);
    if (!obj) return null;
    return await obj.json();
  } catch { return null; }
}

async function bumpShareClick(env, tok, rec) {
  rec.click_count = (rec.click_count || 0) + 1;
  rec.last_seen_at = new Date().toISOString();
  await env.BUCKET.put(`shares/${tok}.json`, JSON.stringify(rec), {
    httpMetadata: { contentType: 'application/json' },
  });
}

async function handleViewShare(request, env, tok) {
  const rec = await loadShareRecord(env, tok);
  if (!rec) {
    return new Response('Share link not found or expired.', { status: 404 });
  }
  // Don't count HEAD as a click (link-preview pings etc.).
  if (request.method === 'GET') {
    request.ctx?.waitUntil?.(bumpShareClick(env, tok, rec));
  }

  const title = rec.vid || 'Tennis Uploader';
  const videoUrl = `/v/${tok}/video`;
  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <meta property="og:title" content="${escapeHtml(title)} — Tennis Uploader" />
  <meta property="og:type" content="video.other" />
  <meta property="og:video" content="https://tennis.playfullife.com${videoUrl}" />
  <meta property="og:video:type" content="video/mp4" />
  <title>${escapeHtml(title)} — Tennis Uploader</title>
  <style>
    :root { color-scheme: dark; }
    html, body { margin:0; padding:0; background:#0A0A0B; color:#F5F5F7;
      font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
    .wrap { max-width: 980px; margin: 0 auto; padding: env(safe-area-inset-top) 0 0; }
    video { width:100%; max-height: 80vh; background:#000; display:block; }
    .meta { padding: 16px 18px; }
    h1 { font-size: 20px; margin: 0 0 6px; letter-spacing:-0.01em; }
    .sub { color:#9AA0A6; font-size:13px; }
    .footer { padding: 14px 18px 28px; color:#9AA0A6; font-size:13px; }
    .footer a { color:#C7FF00; text-decoration: none; font-weight:600; }
  </style>
</head>
<body>
  <div class="wrap">
    <video controls playsinline preload="metadata" src="${videoUrl}"></video>
    <div class="meta">
      <h1>${escapeHtml(title)}</h1>
      <div class="sub">Shared from Tennis Uploader</div>
    </div>
  </div>
  <div class="footer">
    Want your own swing breakdowns?
    <a href="https://apps.apple.com/us/app/tennis-uploader/id6772337106">Get the app</a>.
  </div>
</body>
</html>`;
  return new Response(html, {
    status: 200,
    headers: {
      'content-type': 'text/html; charset=utf-8',
      'cache-control': 'no-store, no-cache, must-revalidate, max-age=0',
      'cdn-cache-control': 'no-store',
      'x-frame-options': 'SAMEORIGIN',
    },
  });
}

async function handleShareMedia(request, env, tok) {
  const rec = await loadShareRecord(env, tok);
  if (!rec) {
    return new Response('Not Found', { status: 404 });
  }
  const key = `highlights/${rec.owner}/${rec.vid}/${rec.vid}_timeline.mp4`;
  return serveR2Object(request, env, key, { private: true });
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// Delete every R2 key associated with one video — gallery outputs,
// thumbnail, source file, marker — covering both the per-user-prefixed
// layout (post-refactor) and the legacy flat layout (in case stragglers
// remain). Returns an array of keys actually deleted. Used by both
// /api/video/:vid/delete (per-video, owner JWT) and /api/account
// (bulk, account deletion).
async function deleteVideoFamily(env, vid, owner) {
  const deleted = [];
  const ownerPrefix = owner ? `${owner}/` : '';
  const prefixesToList = [
    `highlights/${ownerPrefix}${vid}/`,
    `highlights/${vid}/`,
    `processed/${ownerPrefix}${vid}/`,
    `processed/${vid}/`,
  ];
  for (const prefix of prefixesToList) {
    let cursor;
    do {
      const page = await env.BUCKET.list({ prefix, cursor });
      for (const obj of page.objects) {
        await env.BUCKET.delete(obj.key).catch(() => {});
        deleted.push(obj.key);
      }
      cursor = page.truncated ? page.cursor : undefined;
    } while (cursor);
  }
  for (const thumbKey of [
    `highlights/${ownerPrefix}thumbs/${vid}.jpg`,
    `highlights/thumbs/${vid}.jpg`,
    `thumbs/${vid}.jpg`,
  ]) {
    try { await env.BUCKET.delete(thumbKey); deleted.push(thumbKey); } catch {}
  }
  for (const ext of ['mov', 'mp4', 'MOV', 'MP4']) {
    try {
      await env.BUCKET.delete(`source/${vid}.${ext}`);
      deleted.push(`source/${vid}.${ext}`);
    } catch {}
  }
  try {
    await env.BUCKET.delete(`uploads/${vid}.json`);
    deleted.push(`uploads/${vid}.json`);
  } catch {}
  return deleted;
}

async function handleDeleteVideo(request, env, cors, vid) {
  // Sanitize vid: only allow safe characters.
  if (!/^[A-Za-z0-9_-]+$/.test(vid)) {
    return jsonResponse({ error: 'Invalid video id' }, 400, cors);
  }

  // Resolve the owner from the marker. If the marker lacks a user_hash
  // (legacy upload), only an admin can delete.
  let owner = null;
  try {
    const markerObj = await env.BUCKET.get(`uploads/${vid}.json`);
    if (markerObj) {
      const marker = await markerObj.json();
      owner = marker.user_hash || marker.uploaded_by || null;
    }
  } catch {}

  // Authorize. Two paths:
  //   a) JWT cookie/header whose sub matches the marker's owner
  //   b) JWT cookie/header whose sub is an admin
  // The legacy `deletevideo` shared password is gone — every web user
  // now signs in via SIWA or magic-link, both of which carry a JWT
  // cookie that handles delete auth natively.
  const cookieToken = readCookie(request, 'tennis_jwt');
  const headerAuth = (request.headers.get('authorization') || '').startsWith('Bearer ')
    ? request.headers.get('authorization').slice(7).trim() : null;
  const token = cookieToken || headerAuth;
  let claims = null;
  if (token && env.JWT_SIGNING_SECRET) {
    try { claims = await verifyOurJWT(token, env.JWT_SIGNING_SECRET); } catch {}
  }
  const isOwner = !!(claims && owner && claims.sub === owner);
  const isAdmin = !!(claims && isAdminUser(env, claims.sub));

  if (!isOwner && !isAdmin) {
    return jsonResponse(
      { error: 'Not authorized to delete this video' }, 403, cors,
    );
  }

  const deleted = await deleteVideoFamily(env, vid, owner);

  // Append to deletion log.
  let log = [];
  try {
    const logObj = await env.BUCKET.get('highlights/deleted.json');
    if (logObj) log = await logObj.json();
  } catch {}
  log.push({
    video_id: vid,
    deleted_at: new Date().toISOString(),
    files_removed: deleted.length,
    via: isOwner ? 'owner_jwt' : 'admin_jwt',
    by: claims ? claims.sub : null,
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

// Canonical user_hash derivation. Identity is keyed on email so the
// SIWA flow and the magic-link flow deduplicate naturally: same email
// → same hash → same gallery. Keep userHashFromAppleSub for legacy
// callers, but new code should use userHashFromEmail.
async function userHashFromEmail(email) {
  const normalized = (email || '').trim().toLowerCase();
  const data = new TextEncoder().encode(normalized);
  const hash = await crypto.subtle.digest('SHA-256', data);
  const hex = [...new Uint8Array(hash)]
    .slice(0, 4)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  return `u_${hex}`;
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

  // Pull the existing user record (keyed by apple_sub) so we can recover
  // the email Apple sent on first sign-in. Apple only sends `email` on the
  // first request, so we cache it onto the user record and reuse later.
  const userKey = `users/${appleSub}.json`;
  const existingObj = await env.BUCKET.get(userKey);
  const existingRecord = existingObj ? await existingObj.json() : null;
  const isBanned = !!(existingRecord && existingRecord.status === 'banned');

  const email = applePayload.email || (existingRecord && existingRecord.email) || null;
  if (!email) {
    return jsonResponse(
      { error: 'Apple did not return an email and none on file. Sign out + back in on Apple ID to refresh.' },
      400, cors,
    );
  }
  // Identity = sha256(email). SIWA + magic-link share this hash so the
  // same email always resolves to the same gallery.
  const userHash = await userHashFromEmail(email);

  // Allowlist check. Supports `subs` (apple_sub list) AND `emails` (email
  // list) for invite-only mode. Either match grants access. Anyone with an
  // existing user record is grandfathered in.
  try {
    const allowlistObj = await env.BUCKET.get('users/_allowlist.json');
    if (allowlistObj) {
      const allowlist = await allowlistObj.json();
      if (allowlist.open === false) {
        const subs = Array.isArray(allowlist.subs) ? allowlist.subs : [];
        const emails = (allowlist.emails || []).map((e) => e.toLowerCase());
        const isAllowedSub = subs.includes(appleSub);
        const isAllowedEmail = emails.includes(email.toLowerCase());
        if (isBanned) {
          return jsonResponse(
            { error: 'Not approved', detail: 'Your account has been banned by the administrator.' },
            403, cors,
          );
        }
        if (!isAllowedSub && !isAllowedEmail && !existingRecord) {
          return jsonResponse(
            { error: 'Not approved', detail: 'Your account is not on the invite list. Ask the administrator to add you.' },
            403, cors,
          );
        }
      }
    }
  } catch {}

  const nowIso = new Date().toISOString();
  let userRecord;
  if (existingRecord) {
    userRecord = existingRecord;
    userRecord.last_seen = nowIso;
    userRecord.user_hash = userHash;
    if (email && !userRecord.email) userRecord.email = email;
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
      email,
      email_verified:
        applePayload.email_verified === 'true' || applePayload.email_verified === true || null,
      auth_methods: ['apple'],
    };
  }
  // Track that this account has authenticated via apple at least once.
  userRecord.auth_methods = Array.from(new Set([...(userRecord.auth_methods || []), 'apple']));
  await env.BUCKET.put(userKey, JSON.stringify(userRecord), {
    httpMetadata: { contentType: 'application/json' },
  });
  // Maintain an email→user_hash index used by the magic-link flow to
  // resolve incoming sign-ins. Cheap: small JSON per email.
  await writeEmailIndex(env, email, { user_hash: userHash, apple_sub: appleSub });

  const nowSec = Math.floor(Date.now() / 1000);
  const claims = {
    sub: userHash,
    apple_sub: appleSub,
    email,
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

// ---------------------------------------------------------------------------
// Magic-link auth (email-based, no Apple required)
// ---------------------------------------------------------------------------
//
// POST /api/auth/magic/request  { email }
//   - Enforces allowlist (open mode OR email in allowlist OR existing user)
//   - Generates a single-use token (URL-safe random) with 15-min TTL
//   - Stores `magic/<token>.json` { email, exp, used:false }
//   - Sends a one-click link via Resend
//   - Always returns 200 (don't leak which emails are valid)
//
// GET  /api/auth/magic/consume?token=<token>&continue=<url>
//   - Validates token, marks used:true (single-use)
//   - Mints our JWT (sub = email-hash), sets cookie, 302 → /u/<hash>
const MAGIC_TOKEN_TTL_SEC = 15 * 60;

// Per-email rate limit: 30s minimum gap between magic-link requests
// and at most 5 in any 1-hour window. Blocks trivial email-bomb abuse.
const MAGIC_MIN_GAP_SEC = 30;
const MAGIC_HOURLY_LIMIT = 5;

async function magicRateLimit(env, email) {
  const hashBuf = await crypto.subtle.digest(
    'SHA-256', new TextEncoder().encode(email),
  );
  const key = `magic_rate/${[...new Uint8Array(hashBuf)]
    .slice(0, 16).map((b) => b.toString(16).padStart(2, '0')).join('')}.json`;
  let record = { sends: [] };
  try {
    const obj = await env.BUCKET.get(key);
    if (obj) record = await obj.json();
  } catch {}
  const nowSec = Math.floor(Date.now() / 1000);
  // Drop entries older than 1 hour.
  record.sends = (record.sends || []).filter((t) => nowSec - t < 3600);
  const last = record.sends[record.sends.length - 1] || 0;
  if (last && nowSec - last < MAGIC_MIN_GAP_SEC) {
    return { ok: false, reason: 'gap', waitSec: MAGIC_MIN_GAP_SEC - (nowSec - last) };
  }
  if (record.sends.length >= MAGIC_HOURLY_LIMIT) {
    return { ok: false, reason: 'hourly' };
  }
  record.sends.push(nowSec);
  await env.BUCKET.put(key, JSON.stringify(record), {
    httpMetadata: { contentType: 'application/json' },
  });
  return { ok: true };
}

async function handleMagicRequest(request, env, cors) {
  const body = await request.json().catch(() => ({}));
  const rawEmail = (body.email || '').toString().trim().toLowerCase();
  if (!rawEmail || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(rawEmail)) {
    return jsonResponse({ error: 'Valid email required' }, 400, cors);
  }

  // Rate limit BEFORE the allowlist check so an attacker can't probe
  // which emails are allowed by timing the response.
  const rl = await magicRateLimit(env, rawEmail);
  if (!rl.ok) {
    // Always return 200 so we don't leak existence either; just tell
    // the user we're throttling them.
    const message = rl.reason === 'gap'
      ? `Just sent one — try again in ${rl.waitSec || MAGIC_MIN_GAP_SEC}s.`
      : 'Too many sign-in requests. Try again in an hour.';
    return jsonResponse({ ok: true, sent: false, message }, 200, cors);
  }

  // Allowlist (same rules as Apple flow): open mode, email match, or
  // an existing user record indexed by email.
  let isAllowed = true;
  try {
    const allowlistObj = await env.BUCKET.get('users/_allowlist.json');
    if (allowlistObj) {
      const allowlist = await allowlistObj.json();
      if (allowlist.open === false) {
        const emails = (allowlist.emails || []).map((e) => e.toLowerCase());
        const isAllowedEmail = emails.includes(rawEmail);
        const existing = await readEmailIndex(env, rawEmail);
        isAllowed = isAllowedEmail || !!existing;
      }
    }
  } catch {}
  if (!isAllowed) {
    // Always 200 — don't leak allowlist contents. The user just won't get
    // an email. We log the denial for the operator.
    console.log(`magic-link denied (not allowed): ${rawEmail}`);
    return jsonResponse({ ok: true, sent: false, message: 'If this email is invited, a sign-in link is on the way.' }, 200, cors);
  }

  const token = base64urlEncodeBytes(crypto.getRandomValues(new Uint8Array(32)));
  const nowSec = Math.floor(Date.now() / 1000);
  const record = {
    email: rawEmail,
    iat: nowSec,
    exp: nowSec + MAGIC_TOKEN_TTL_SEC,
    used: false,
  };
  await env.BUCKET.put(`magic/${token}.json`, JSON.stringify(record), {
    httpMetadata: { contentType: 'application/json' },
  });

  const link = `https://tennis.playfullife.com/api/auth/magic/consume?token=${token}`;
  try {
    await sendMagicLinkEmail(env, rawEmail, link);
  } catch (e) {
    console.log(`magic-link send failed for ${rawEmail}: ${e.message}`);
    return jsonResponse(
      { error: 'Email send failed', detail: 'Try again or contact the administrator.' },
      500, cors,
    );
  }
  return jsonResponse({ ok: true, sent: true, message: 'Check your inbox for a sign-in link.' }, 200, cors);
}

async function handleMagicConsume(request, env, cors) {
  const url = new URL(request.url);
  const token = url.searchParams.get('token') || '';
  if (!token || !/^[A-Za-z0-9_-]+$/.test(token)) {
    return new Response('Invalid sign-in link.', { status: 400 });
  }
  const key = `magic/${token}.json`;
  const obj = await env.BUCKET.get(key);
  if (!obj) {
    return new Response('Sign-in link not found or already used.', { status: 404 });
  }
  const record = await obj.json();
  const nowSec = Math.floor(Date.now() / 1000);
  if (record.used) {
    return new Response('Sign-in link already used. Request a new one.', { status: 410 });
  }
  if (record.exp < nowSec) {
    return new Response('Sign-in link expired. Request a new one.', { status: 410 });
  }

  // Burn the token first (so refresh-double-click doesn't replay).
  record.used = true;
  record.used_at = new Date().toISOString();
  await env.BUCKET.put(key, JSON.stringify(record), {
    httpMetadata: { contentType: 'application/json' },
  });

  const email = record.email;
  const userHash = await userHashFromEmail(email);

  // Find-or-create a user record. For magic-link-only users we key the
  // user record on a synthetic `magic:<email>` instead of an apple_sub.
  const existingByEmail = await readEmailIndex(env, email);
  let userKey;
  let userRecord;
  if (existingByEmail && existingByEmail.apple_sub) {
    // Email was previously associated with an Apple account — reuse it.
    userKey = `users/${existingByEmail.apple_sub}.json`;
    const ex = await env.BUCKET.get(userKey);
    userRecord = ex ? await ex.json() : null;
  }
  if (!userRecord) {
    userKey = `users/magic:${email}.json`;
    const ex = await env.BUCKET.get(userKey);
    userRecord = ex ? await ex.json() : null;
  }
  const nowIso = new Date().toISOString();
  if (userRecord) {
    if (userRecord.status === 'banned') {
      return new Response('Your account has been banned by the administrator.', { status: 403 });
    }
    userRecord.last_seen = nowIso;
    userRecord.user_hash = userHash;
    if (userRecord.deleted_at) {
      delete userRecord.deleted_at;
      userRecord.status = 'active';
      userRecord.rejoined_at = nowIso;
    }
  } else {
    userRecord = {
      email,
      user_hash: userHash,
      created_at: nowIso,
      last_seen: nowIso,
      video_count: 0,
      status: 'active',
      auth_methods: ['magic'],
    };
  }
  userRecord.auth_methods = Array.from(new Set([...(userRecord.auth_methods || []), 'magic']));
  await env.BUCKET.put(userKey, JSON.stringify(userRecord), {
    httpMetadata: { contentType: 'application/json' },
  });
  await writeEmailIndex(env, email, {
    user_hash: userHash,
    apple_sub: userRecord.apple_sub || null,
  });

  const claims = {
    sub: userHash,
    email,
    iat: nowSec,
    exp: nowSec + OUR_JWT_TTL_SEC,
    scope: 'upload',
  };
  if (userRecord.apple_sub) claims.apple_sub = userRecord.apple_sub;
  const jwt = await signOurJWT(claims, env.JWT_SIGNING_SECRET);

  return new Response(null, {
    status: 302,
    headers: {
      'location': `/u/${userHash}`,
      'set-cookie': `tennis_jwt=${jwt}; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age=${OUR_JWT_TTL_SEC}`,
    },
  });
}

// ---------------------------------------------------------------------------
// Email sending — Resend
// ---------------------------------------------------------------------------
async function sendMagicLinkEmail(env, toEmail, link) {
  if (!env.RESEND_API_KEY) {
    throw new Error('RESEND_API_KEY not configured');
  }
  const from = env.MAGIC_LINK_FROM || 'Tennis Uploader <onboarding@resend.dev>';
  const subject = 'Sign in to Tennis Uploader';
  const html = `<!doctype html><html><body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;color:#1c1c1e;max-width:520px;margin:32px auto;padding:24px;">
    <h1 style="font-size:22px;margin:0 0 10px;">Sign in to Tennis Uploader</h1>
    <p style="font-size:15px;line-height:1.5;color:#3a3a3c;">Click the button below to sign in. The link works once and expires in 15 minutes.</p>
    <p style="margin:24px 0;"><a href="${link}" style="display:inline-block;padding:12px 22px;border-radius:10px;background:#0a84ff;color:#fff;text-decoration:none;font-weight:600;">Sign in</a></p>
    <p style="font-size:12px;color:#6b7280;">If you didn't request this, you can safely ignore this email.</p>
    <p style="font-size:12px;color:#6b7280;word-break:break-all;">Or copy this link: ${link}</p>
  </body></html>`;
  const text = `Sign in to Tennis Uploader\n\nOpen this link (works once, expires in 15 min):\n${link}\n\nIf you didn't request this, ignore this email.\n`;

  const resp = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      'authorization': `Bearer ${env.RESEND_API_KEY}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({ from, to: [toEmail], subject, html, text }),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Resend ${resp.status}: ${body.slice(0, 300)}`);
  }
}

// users/_email_index/<sha-of-email>.json — maps email → user record refs.
// Used by the magic-link flow and any future "find me by email" code.
async function writeEmailIndex(env, email, refs) {
  const normalized = (email || '').trim().toLowerCase();
  if (!normalized) return;
  const hashBuf = await crypto.subtle.digest(
    'SHA-256', new TextEncoder().encode(normalized),
  );
  const key = [...new Uint8Array(hashBuf)]
    .slice(0, 16).map((b) => b.toString(16).padStart(2, '0')).join('');
  const payload = {
    email: normalized,
    updated_at: new Date().toISOString(),
    ...refs,
  };
  await env.BUCKET.put(
    `users/_email_index/${key}.json`,
    JSON.stringify(payload),
    { httpMetadata: { contentType: 'application/json' } },
  );
}

async function readEmailIndex(env, email) {
  const normalized = (email || '').trim().toLowerCase();
  if (!normalized) return null;
  const hashBuf = await crypto.subtle.digest(
    'SHA-256', new TextEncoder().encode(normalized),
  );
  const key = [...new Uint8Array(hashBuf)]
    .slice(0, 16).map((b) => b.toString(16).padStart(2, '0')).join('');
  try {
    const obj = await env.BUCKET.get(`users/_email_index/${key}.json`);
    return obj ? await obj.json() : null;
  } catch { return null; }
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

  // Find every marker tagged with this user_hash (the new `user_hash`
  // field or the legacy `uploaded_by`) and delete the full file family.
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
        const ownerOnMarker = marker.user_hash || marker.uploaded_by;
        if (ownerOnMarker !== userHash) continue;
        await deleteVideoFamily(env, marker.video_id, userHash);
        deletedVideos.push(marker.video_id);
      } catch {}
    }
    cursor = listed.truncated ? listed.cursor : undefined;
  } while (cursor);
  // The user's per-user gallery index is now stale — drop it so we don't
  // serve a 200 for `/u/<hash>` to a future visitor.
  try { await env.BUCKET.delete(`highlights/${userHash}/index.html`); } catch {}

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
