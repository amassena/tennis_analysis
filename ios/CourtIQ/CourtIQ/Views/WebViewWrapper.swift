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
    /// One-shot JS snippet the parent SwiftUI can hand down to the
    /// WebView (e.g. `applyNativeFilter({filter:"all",sort:"recorded-desc"})`).
    /// When non-nil, we evaluate it and reset the binding to nil so the
    /// same script doesn't replay on every SwiftUI redraw.
    @Binding var pendingScript: String?

    init(url: URL, pendingScript: Binding<String?> = .constant(nil)) {
        self.url = url
        self._pendingScript = pendingScript
    }

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
        // WKWebView silently no-ops alert/confirm/prompt unless we
        // implement WKUIDelegate. The gallery's deleteVideo() relies on
        // confirm() to gate the destructive POST — without this it
        // appears to do nothing on phone.
        webView.uiDelegate = context.coordinator
        // PR-L: disable WKWebView's left-edge swipe-back so it can't
        // collide with horizontal filmstrip / sequence-strip scrolling
        // inside the gallery.
        webView.allowsBackForwardNavigationGestures = false
        seedAuthCookieAndLoad(webView: webView, url: url)
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        if uiView.url != url {
            seedAuthCookieAndLoad(webView: uiView, url: url)
        }
        if let script = pendingScript, !script.isEmpty {
            uiView.evaluateJavaScript(script, completionHandler: nil)
            DispatchQueue.main.async { self.pendingScript = nil }
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

    class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler, WKUIDelegate {
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

        /// Last WebView we received a message from. Saved so the native
        /// CoachSummarySheet can call back into the WebView's JS
        /// (jumpToExample) when the user taps an example timestamp.
        weak var lastWebView: WKWebView?

        func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage,
        ) {
            lastWebView = message.webView
            switch message.name {
            case "openVideo":
                guard let body = message.body as? [String: Any],
                      let urlStr = body["url"] as? String,
                      let url = URL(string: urlStr) else { return }
                let title = body["title"] as? String ?? ""
                let startTime = (body["startTime"] as? NSNumber)?.doubleValue
                    ?? (body["startTime"] as? Double)
                let videoId = body["videoId"] as? String
                let variant = body["variant"] as? String
                presentNativePlayer(
                    url: url, title: title,
                    videoId: videoId, variant: variant,
                    startTime: startTime,
                )
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
            guard JSONSerialization.isValidJSONObject(coachingRaw),
                  let data = try? JSONSerialization.data(withJSONObject: coachingRaw),
                  let payload = try? JSONDecoder().decode(CoachPayload.self, from: data)
            else { return }
            guard let top = topPresentedVC() else { return }
            // Capture the WebView at presentation-time so the chip-tap
            // closure can dispatch JS back into the gallery.
            let webView = lastWebView
            let host = UIHostingController(
                rootView: CoachSummarySheet(
                    videoId: videoId,
                    payload: payload,
                    onTapExample: { t in
                        let escaped = videoId.replacingOccurrences(of: "'", with: "\\'")
                        webView?.evaluateJavaScript(
                            "jumpToExample('\(escaped)', \(t))",
                            completionHandler: nil,
                        )
                    },
                ),
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

        // MARK: WKUIDelegate — bridge JS alert/confirm/prompt to native UIAlerts

        func webView(
            _ webView: WKWebView,
            runJavaScriptAlertPanelWithMessage message: String,
            initiatedByFrame frame: WKFrameInfo,
            completionHandler: @escaping () -> Void,
        ) {
            guard let top = topPresentedVC() else { completionHandler(); return }
            let a = UIAlertController(title: nil, message: message, preferredStyle: .alert)
            a.addAction(UIAlertAction(title: "OK", style: .default) { _ in completionHandler() })
            top.present(a, animated: true)
        }

        func webView(
            _ webView: WKWebView,
            runJavaScriptConfirmPanelWithMessage message: String,
            initiatedByFrame frame: WKFrameInfo,
            completionHandler: @escaping (Bool) -> Void,
        ) {
            guard let top = topPresentedVC() else { completionHandler(false); return }
            let a = UIAlertController(title: nil, message: message, preferredStyle: .alert)
            a.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in completionHandler(false) })
            a.addAction(UIAlertAction(title: "OK", style: .destructive) { _ in completionHandler(true) })
            top.present(a, animated: true)
        }

        func webView(
            _ webView: WKWebView,
            runJavaScriptTextInputPanelWithPrompt prompt: String,
            defaultText: String?,
            initiatedByFrame frame: WKFrameInfo,
            completionHandler: @escaping (String?) -> Void,
        ) {
            guard let top = topPresentedVC() else { completionHandler(nil); return }
            let a = UIAlertController(title: nil, message: prompt, preferredStyle: .alert)
            a.addTextField { tf in tf.text = defaultText }
            a.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in completionHandler(nil) })
            a.addAction(UIAlertAction(title: "OK", style: .default) { _ in
                completionHandler(a.textFields?.first?.text)
            })
            top.present(a, animated: true)
        }

        private func presentNativePlayer(
            url: URL, title: String,
            videoId: String? = nil, variant: String? = nil,
            startTime: Double? = nil,
        ) {
            guard let top = topPresentedVC() else { return }
            // Phase 3: wrap AVPlayerViewController in FilterablePlayerView
            // so the chip filter row + slo-mo toggle live in native too.
            // The SwiftUI view owns the AVPlayer lifecycle, fetches
            // shots.json with the user's JWT, and seeks past gaps when a
            // non-'all' filter is active.
            //
            // PortraitHostingController locks the view to portrait so
            // device rotation doesn't trigger AVPlayerViewController's
            // system landscape-fullscreen takeover (which would drop the
            // chip overlay). User can still tap the corner ⤢ button to
            // explicitly request fullscreen.
            let host = PortraitHostingController(
                rootView: FilterablePlayerView(
                    url: url,
                    title: title,
                    videoId: videoId,
                    variant: variant,
                    startTime: startTime,
                ),
            )
            host.modalPresentationStyle = .fullScreen
            host.view.backgroundColor = .black
            top.present(host, animated: true)
        }
    }
}

/// Hosts FilterablePlayerView. Since build 19 the player uses a custom
/// AVPlayerLayer (not AVPlayerViewController), there's no system
/// fullscreen takeover to defend against — so this hosting controller
/// allows landscape rotation. The SwiftUI body re-flows naturally,
/// and a dedicated fullscreen-toggle button in the custom controls
/// owns the "fill the screen" experience.
final class PortraitHostingController<Content: View>: UIHostingController<Content> {
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        .allButUpsideDown
    }
    override var shouldAutorotate: Bool { true }
}
