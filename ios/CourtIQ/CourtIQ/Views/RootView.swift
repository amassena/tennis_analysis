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
            UploadTabPlaceholder(userHash: userHash)
                .tabItem {
                    Label("Upload", systemImage: "icloud.and.arrow.up")
                }

            GalleryTabPlaceholder()
                .tabItem {
                    Label("Gallery", systemImage: "play.rectangle.on.rectangle")
                }
        }
    }
}

private struct UploadTabPlaceholder: View {
    let userHash: String
    @EnvironmentObject var auth: AuthCoordinator

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "icloud.and.arrow.up")
                .font(.system(size: 56))
                .foregroundColor(.secondary)
            Text("Upload")
                .font(.title2.bold())
            Text("Signed in as \(userHash). Real upload flow lands in PR 2+.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            Button("Sign out") { auth.signOut() }
                .padding(.top, 24)
        }
    }
}

private struct GalleryTabPlaceholder: View {
    var body: some View {
        WebViewWrapper(url: URL(string: "https://tennis.playfullife.com")!)
            .ignoresSafeArea(edges: .bottom)
    }
}
