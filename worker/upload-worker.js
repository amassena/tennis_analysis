/**
 * Tennis Media Upload Worker
 *
 * Handles chunked video uploads to R2 via multipart upload API.
 * Deployed on media.playfullife.com/api/* — all other routes fall
 * through to R2 static file serving (custom domain).
 *
 * Routes:
 *   POST /api/upload/init       — validate password, start multipart upload
 *   PUT  /api/upload/:id/:part  — upload a chunk
 *   POST /api/upload/:id/complete — finalize upload
 *   POST /api/upload/link       — submit iCloud share link (no file upload)
 *   GET  /api/status/:id        — check processing status
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    const cors = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: cors });
    }

    try {
      // POST /api/upload/init
      if (path === '/api/upload/init' && request.method === 'POST') {
        return await handleInit(request, env, cors);
      }

      // PUT /api/upload/:id/:partNumber
      const partMatch = path.match(/^\/api\/upload\/([^/]+)\/(\d+)$/);
      if (partMatch && request.method === 'PUT') {
        return await handlePart(request, env, cors, partMatch[1], parseInt(partMatch[2]));
      }

      // POST /api/upload/link
      if (path === '/api/upload/link' && request.method === 'POST') {
        return await handleLink(request, env, cors);
      }

      // POST /api/upload/:id/complete
      const completeMatch = path.match(/^\/api\/upload\/([^/]+)\/complete$/);
      if (completeMatch && request.method === 'POST') {
        return await handleComplete(request, env, cors, completeMatch[1]);
      }

      // GET /api/status/:id
      const statusMatch = path.match(/^\/api\/status\/([^/]+)$/);
      if (statusMatch && request.method === 'GET') {
        return await handleStatus(env, cors, statusMatch[1]);
      }

      return jsonResponse({ error: 'Not found' }, 404, cors);
    } catch (err) {
      return jsonResponse({ error: err.message }, 500, cors);
    }
  },
};

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

  // Start R2 multipart upload
  const multipart = await env.BUCKET.createMultipartUpload(key);

  // Store metadata alongside video
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

  return jsonResponse({ partNumber: part.partNumber, etag: part.etag }, 200, cors);
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

  // Update metadata: ready for Mac watcher to pick up
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

async function handleStatus(env, cors, id) {
  const metaObj = await env.BUCKET.get(`uploads/${id}.json`);
  if (metaObj) {
    return jsonResponse(await metaObj.json(), 200, cors);
  }
  return jsonResponse({ id, status: 'not_found' }, 404, cors);
}

function generateId() {
  const ts = Date.now().toString(36);
  const rand = Math.random().toString(36).substring(2, 8);
  return `${ts}_${rand}`;
}

function jsonResponse(data, status = 200, headers = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}
