import SwiftUI

struct UploadTabView: View {
    let userHash: String
    @ObservedObject private var manager = UploadManager.shared
    @StateObject private var recent: RecentUploadsModel
    @EnvironmentObject var auth: AuthCoordinator
    @EnvironmentObject var nav: AppNavigation
    @State private var showingComposer = false
    @State private var showingSettings = false

    init(userHash: String) {
        self.userHash = userHash
        _recent = StateObject(wrappedValue: RecentUploadsModel(userHash: userHash))
    }

    var body: some View {
        NavigationView {
            Group {
                if manager.uploads.isEmpty && recent.items.isEmpty {
                    EmptyUploadsView { showingComposer = true }
                } else {
                    List {
                        if !manager.uploads.isEmpty {
                            Section("Uploading") {
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
                        RecentUploadsSection(model: recent) { item in
                            nav.openGallery(anchor: item.video_id)
                        }
                    }
                    .refreshable { await recent.refresh() }
                }
            }
            .onAppear { recent.start() }
            .onDisappear { recent.stop() }
            .navigationTitle("Upload")
            .navigationBarTitleDisplayMode(.inline)
            .safeAreaInset(edge: .bottom) {
                Text(appVersionString)
                    .font(.caption2.monospaced())
                    .foregroundColor(.secondary)
                    .padding(.bottom, 4)
            }
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

    private var appVersionString: String {
        let v = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "—"
        let b = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "—"
        return "v\(v) (\(b))"
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
