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
                        TodayHeroCard(userHash: userHash, recent: recent)
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
                        RecentUploadsSection(userHash: userHash, model: recent) { item in
                            nav.openGallery(anchor: item.video_id)
                        }
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                    .background(Color.brandBackground)
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
        ZStack {
            Color.brandBackground.ignoresSafeArea()
            VStack(spacing: 22) {
                // Neon tennis-ball glyph in a soft halo. Keeps the
                // empty-state on-brand without an extra asset.
                ZStack {
                    Circle()
                        .fill(Color.brandAccent.opacity(0.12))
                        .frame(width: 168, height: 168)
                    Circle()
                        .fill(Color.brandAccent)
                        .frame(width: 96, height: 96)
                    Image(systemName: "figure.tennis")
                        .font(.system(size: 44, weight: .bold))
                        .foregroundColor(.brandBackground)
                }
                VStack(spacing: 8) {
                    Text("Your gallery is empty")
                        .font(.title2.weight(.bold))
                        .foregroundColor(.brandText)
                    Text("Record a session or pick a clip from Photos.\nWe'll detect every shot and stack it against the pros.")
                        .font(.subheadline)
                        .foregroundColor(.brandTextSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }
                Button(action: onAdd) {
                    Label("Upload your first video", systemImage: "plus.circle.fill")
                        .font(.headline)
                        .frame(maxWidth: 320, minHeight: 50)
                }
                .buttonStyle(.borderedProminent)
                .tint(.brandAccent)
                .foregroundColor(.brandBackground)
                .padding(.top, 4)
            }
            .padding(.bottom, 40)
        }
    }
}
