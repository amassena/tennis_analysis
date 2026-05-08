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
 *   GET  /api/upload/iphone/check  → idempotency probe (?asset_id=…)
 *   GET  /api/status/:id           → check processing status
 *   POST /api/status/:id/update    → update processing status (auth)
 *   GET  /api/queue                → list all uploads and their status
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
  } else {
    key = path.slice(1); // strip leading /
  }

  if (request.method === 'HEAD') {
    return handleHead(env, key);
  }

  // GET — supports range requests for video streaming (skip for downloads)
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

  // Fallback: try highlights/ prefix for clean URLs (e.g. /IMG_1108/file.mp4)
  if (!obj && !key.startsWith('highlights/') && !key.startsWith('uploads/')) {
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

  // Force download when ?dl=1 is present
  if (isDownload) {
    const filename = key.split('/').pop();
    headers.set('content-disposition', `attachment; filename="${filename}"`);
  }

  // Cache: videos 24h, html always revalidate, images 1h
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

  // Conditional request not met → 304
  if (!obj.body) {
    return new Response(null, { status: 304, headers });
  }

  // Range response → 206 only when the client actually sent a Range header
  // (R2 sometimes populates obj.range anyway; returning 206 to a non-range
  // request breaks <img> tags and download managers)
  if (obj.range && rangeHeader && !isDownload) {
    const { offset, length } = obj.range;
    headers.set(
      'content-range',
      `bytes ${offset}-${offset + length - 1}/${obj.size}`
    );
    headers.set('content-length', String(length));
    return new Response(obj.body, { status: 206, headers });
  }

  // Full response → 200
  headers.set('content-length', String(obj.size));
  // Ensure cache-control is set (explicitly set after all other header manipulation)
  if (isHtml) {
    headers.set('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
    headers.set('Pragma', 'no-cache');
    headers.set('Expires', '0');
  }
  return new Response(obj.body, { status: 200, headers });
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
    'Access-Control-Allow-Methods': 'GET, POST, PUT, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };
}

async function handleApi(request, env, path) {
  const cors = corsHeaders();

  try {
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
  const auth = request.headers.get('authorization') || '';
  const expected = env.IPHONE_UPLOAD_TOKEN;
  if (!expected || auth !== `Bearer ${expected}`) {
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
