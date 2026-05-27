import SwiftUI
import WebKit
import AVKit

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
        // JS bridges. Gallery JS posts to these via
        // window.webkit.messageHandlers.<name>.postMessage(payload):
        //   openVideo → hand the URL off to AVPlayerViewController
        //   openCoach → present a native SwiftUI CoachSummarySheet
        config.userContentController.add(context.coordinator, name: "openVideo")
        config.userContentController.add(context.coordinator, name: "openCoach")

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

    class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
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

        func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage,
        ) {
            switch message.name {
            case "openVideo":
                guard let body = message.body as? [String: Any],
                      let urlStr = body["url"] as? String,
                      let url = URL(string: urlStr) else { return }
                let title = body["title"] as? String ?? ""
                presentNativePlayer(url: url, title: title)
            case "openCoach":
                guard let body = message.body as? [String: Any],
                      let vid = body["vid"] as? String,
                      let coaching = body["coaching"] else { return }
                presentNativeCoach(videoId: vid, coachingRaw: coaching)
            default:
                break
            }
        }

        private func presentNativeCoach(videoId: String, coachingRaw: Any) {
            // Re-encode the JS-side payload to JSON, then decode as our
            // strongly-typed CoachPayload. This is the cleanest bridge —
            // works regardless of how JSONSerialization typed the nested
            // values.
            guard JSONSerialization.isValidJSONObject(coachingRaw),
                  let data = try? JSONSerialization.data(withJSONObject: coachingRaw),
                  let payload = try? JSONDecoder().decode(CoachPayload.self, from: data)
            else { return }
            guard let top = topPresentedVC() else { return }
            let host = UIHostingController(
                rootView: CoachSummarySheet(videoId: videoId, payload: payload),
            )
            host.modalPresentationStyle = .pageSheet
            if let sheet = host.sheetPresentationController {
                sheet.detents = [.medium(), .large()]
                sheet.prefersGrabberVisible = true
            }
            top.present(host, animated: true)
        }

        private func topPresentedVC() -> UIViewController? {
            guard let scene = UIApplication.shared.connectedScenes
                .first(where: { $0.activationState == .foregroundActive }) as? UIWindowScene,
                  let root = scene.windows.first(where: { $0.isKeyWindow })?.rootViewController
            else { return nil }
            var top = root
            while let presented = top.presentedViewController { top = presented }
            return top
        }

        private func presentNativePlayer(url: URL, title: String) {
            guard let top = topPresentedVC() else { return }
            let player = AVPlayer(url: url)
            let vc = AVPlayerViewController()
            vc.player = player
            vc.modalPresentationStyle = .fullScreen
            vc.allowsPictureInPicturePlayback = true
            top.present(vc, animated: true) { player.play() }
        }
    }
}
