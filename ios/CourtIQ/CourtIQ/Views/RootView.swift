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

    var body: some View {
        TabView(selection: $nav.selectedTab) {
            UploadTabView(userHash: userHash)
                .tabItem {
                    Label("Upload", systemImage: "icloud.and.arrow.up")
                }
                .tag(AppNavigation.Tab.upload)

            GalleryTabView()
                .tabItem {
                    Label("Gallery", systemImage: "play.rectangle.on.rectangle")
                }
                .tag(AppNavigation.Tab.gallery)
        }
    }
}

private struct GalleryTabView: View {
    @EnvironmentObject var nav: AppNavigation

    var body: some View {
        WebViewWrapper(url: galleryURL)
            .ignoresSafeArea(edges: .bottom)
            .onChange(of: nav.selectedTab) { newValue in
                // Anchor consumed once we land on the gallery tab.
                if newValue == .gallery && nav.pendingGalleryAnchor != nil {
                    // Defer clearing so URL update propagates first.
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                        nav.pendingGalleryAnchor = nil
                    }
                }
            }
    }

    private var galleryURL: URL {
        if let anchor = nav.pendingGalleryAnchor {
            return URL(string: "https://tennis.playfullife.com/#\(anchor)")!
        }
        return URL(string: "https://tennis.playfullife.com")!
    }
}
