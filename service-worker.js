const CACHE_NAME = 'triangulum-v3';
const ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  '/icon-maskable.png',
  '/apple-touch-icon.png',
  '/favicon.ico'
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(ASSETS)));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
    ))
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  
  // Never cache API calls or non-GET
  if (e.request.method !== 'GET' || url.hostname.includes('railway.app') || url.hostname.includes('unpkg.com')) {
    return;
  }
  
  // Only cache same-origin static assets
  if (url.origin === location.origin) {
    e.respondWith(
      caches.match(e.request).then(r => r || fetch(e.request))
    );
  }
});