import SwiftUI

struct UploadTabView: View {
    let userHash: String
    @ObservedObject private var manager = UploadManager.shared
    @EnvironmentObject var auth: AuthCoordinator
    @EnvironmentObject var nav: AppNavigation
    @State private var showingComposer = false
    @State private var showingSettings = false

    var body: some View {
        NavigationView {
            Group {
                if manager.uploads.isEmpty {
                    EmptyUploadsView { showingComposer = true }
                } else {
                    List {
                        ForEach(manager.uploads) { upload in
                            UploadRowView(
                                state: upload,
                                onRetry: { manager.retry(id: upload.id) },
                                onDiscard: { manager.discard(id: upload.id) },
                                onViewInGallery: { nav.openGallery(anchor: upload.uploadId) }
                            )
                        }
                    }
                }
            }
            .navigationTitle("Upload")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button { showingSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { showingComposer = true } label: {
                        Image(systemName: "plus.circle.fill")
                            .font(.title3)
                    }
                }
            }
        }
        .sheet(isPresented: $showingComposer) {
            UploadComposerSheet(userHash: userHash, isPresented: $showingComposer)
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView(userHash: userHash, isPresented: $showingSettings)
        }
    }
}

private struct EmptyUploadsView: View {
    var onAdd: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "icloud.and.arrow.up")
                .font(.system(size: 56))
                .foregroundColor(.secondary)
            Text("No uploads yet")
                .font(.title3.weight(.semibold))
            Text("Tap + to record or pick a video.")
                .font(.subheadline)
                .foregroundColor(.secondary)
            Button(action: onAdd) {
                Label("Upload a video", systemImage: "plus")
                    .font(.headline)
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 8)
        }
    }
}
