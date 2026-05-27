import SwiftUI
import WebKit

/// WKWebView wrapper. Reloads with a new URL when `url` changes
/// (used by the Gallery tab to deep-link to `#<video_id>` anchors).
///
/// Auth cookie: when the URL targets tennis.playfullife.com, we inject
/// the user's JWT as a `tennis_jwt` cookie into WKHTTPCookieStore
/// BEFORE loading. The worker also sets the same cookie via 302
/// Set-Cookie on `/u/<hash>?t=<jwt>` requests, but WKWebView is known
/// to drop cookies set during redirects when later subresources (img
/// src, video src) try to fetch — those subresource requests can race
/// the cookie store update and end up unauthenticated. Pre-seeding the
/// store fixes that.
struct WebViewWrapper: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = .black
        webView.scrollView.backgroundColor = .black
        webView.navigationDelegate = context.coordinator
        seedAuthCookieAndLoad(webView: webView, url: url)
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        if uiView.url != url {
            seedAuthCookieAndLoad(webView: uiView, url: url)
        }
    }

    private func seedAuthCookieAndLoad(webView: WKWebView, url: URL) {
        guard let jwt = TokenStore.load(),
              let host = url.host,
              host.contains("playfullife.com")
        else {
            webView.load(URLRequest(url: url))
            return
        }
        let props: [HTTPCookiePropertyKey: Any] = [
            .domain: host,
            .path: "/",
            .name: "tennis_jwt",
            .value: jwt,
            .secure: "TRUE",
            .expires: Date(timeIntervalSinceNow: 30 * 24 * 3600),
        ]
        if let cookie = HTTPCookie(properties: props) {
            webView.configuration.websiteDataStore.httpCookieStore.setCookie(cookie) {
                webView.load(URLRequest(url: url))
            }
        } else {
            webView.load(URLRequest(url: url))
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    class Coordinator: NSObject, WKNavigationDelegate {
        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            if let host = navigationAction.request.url?.host,
               host.contains("playfullife.com") || host.contains("localhost") {
                decisionHandler(.allow)
            } else if navigationAction.navigationType == .linkActivated,
                      let url = navigationAction.request.url {
                UIApplication.shared.open(url)
                decisionHandler(.cancel)
            } else {
                decisionHandler(.allow)
            }
        }
    }
}
