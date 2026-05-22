import SwiftUI

struct UploadTabView: View {
    let userHash: String
    @ObservedObject private var manager = UploadManager.shared
    @EnvironmentObject var auth: AuthCoordinator
    @State private var showingComposer = false

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
                                onDiscard: { manager.discard(id: upload.id) }
                            )
                        }
                    }
                }
            }
            .navigationTitle("Upload")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Menu {
                        Button("Sign out", role: .destructive) { auth.signOut() }
                    } label: {
                        Image(systemName: "person.crop.circle")
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
