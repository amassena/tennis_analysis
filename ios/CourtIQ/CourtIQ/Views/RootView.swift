import SwiftUI

struct RootView: View {
    @EnvironmentObject var auth: AuthCoordinator
    @EnvironmentObject var nav: AppNavigation

    var body: some View {
        Group {
            switch auth.state {
            case .unknown:
                LoadingScreen()
            case .signedOut:
                AuthGateView()
            case .signedIn(let userHash):
                SignedInTabs(userHash: userHash)
            }
        }
        .task {
            if auth.state == .unknown {
                await auth.bootstrap()
            }
        }
    }
}

private struct LoadingScreen: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            ProgressView().tint(.white)
        }
    }
}

private struct SignedInTabs: View {
    let userHash: String
    @EnvironmentObject var nav: AppNavigation
    @State private var showingWelcome = !WelcomeFlag.didShow

    var body: some View {
        TabView(selection: $nav.selectedTab) {
            UploadTabView(userHash: userHash)
                .tabItem {
                    Label("Upload", systemImage: "icloud.and.arrow.up")
                }
                .tag(AppNavigation.Tab.upload)

            GalleryTabView(userHash: userHash)
                .tabItem {
                    Label("Gallery", systemImage: "play.rectangle.on.rectangle")
                }
                .tag(AppNavigation.Tab.gallery)
        }
        .sheet(isPresented: $showingWelcome, onDismiss: { WelcomeFlag.didShow = true }) {
            WelcomeSheet(isPresented: $showingWelcome, userHash: userHash)
        }
    }
}

private struct GalleryTabView: View {
    let userHash: String
    @EnvironmentObject var nav: AppNavigation
    @State private var filter = FilterState()
    @State private var pendingScript: String?
    @State private var showingSettings = false

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                GalleryFilterBar(state: $filter) { newState in
                    pendingScript = "applyNativeFilter(\(newState.toJSObject()))"
                }
                WebViewWrapper(url: galleryURL, pendingScript: $pendingScript)
                    .ignoresSafeArea(edges: .bottom)
            }
            .background(Color.brandBackground)
            .navigationTitle("Gallery")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button { showingSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView(userHash: userHash, isPresented: $showingSettings)
        }
        .onChange(of: nav.selectedTab) { newValue in
            if newValue == .gallery && nav.pendingGalleryAnchor != nil {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    nav.pendingGalleryAnchor = nil
                }
            }
        }
    }

    // Per-user gallery URL. Always carries the JWT as `?t=` so the
    // worker can re-set the auth cookie even if WKWebView cleared it
    // between launches. The worker 302s back to the clean URL after
    // setting Set-Cookie.
    private var galleryURL: URL {
        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "tennis.playfullife.com"
        comps.path = "/u/\(userHash)"
        if let jwt = TokenStore.load() {
            comps.queryItems = [URLQueryItem(name: "t", value: jwt)]
        }
        if let anchor = nav.pendingGalleryAnchor {
            comps.fragment = anchor
        }
        return comps.url ?? URL(string: "https://tennis.playfullife.com/u/\(userHash)")!
    }
}
