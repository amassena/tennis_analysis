import SwiftUI

struct RootView: View {
    @EnvironmentObject var auth: AuthCoordinator

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
            ProgressView()
                .tint(.white)
        }
    }
}

private struct SignedInTabs: View {
    let userHash: String

    var body: some View {
        TabView {
            UploadTabView(userHash: userHash)
                .tabItem {
                    Label("Upload", systemImage: "icloud.and.arrow.up")
                }

            GalleryTabView()
                .tabItem {
                    Label("Gallery", systemImage: "play.rectangle.on.rectangle")
                }
        }
    }
}

private struct GalleryTabView: View {
    var body: some View {
        WebViewWrapper(url: URL(string: "https://tennis.playfullife.com")!)
            .ignoresSafeArea(edges: .bottom)
    }
}
