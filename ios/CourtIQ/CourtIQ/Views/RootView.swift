import SwiftUI

struct RootView: View {
    var body: some View {
        TabView {
            UploadTabPlaceholder()
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
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "icloud.and.arrow.up")
                .font(.system(size: 56))
                .foregroundColor(.secondary)
            Text("Upload")
                .font(.title2.bold())
            Text("Sign-in and uploads land here in PR 1+.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
    }
}

private struct GalleryTabPlaceholder: View {
    var body: some View {
        WebViewWrapper(url: URL(string: "https://tennis.playfullife.com")!)
            .ignoresSafeArea(edges: .bottom)
    }
}
