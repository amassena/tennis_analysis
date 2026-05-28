import SwiftUI

/// The "+ → Record or Pick" choice screen shown from the Upload tab.
struct UploadComposerSheet: View {
    let userHash: String
    @Binding var isPresented: Bool

    @State private var showingPicker = false
    @State private var showingRecord = false

    var body: some View {
        NavigationView {
            VStack(spacing: 18) {
                Text("Upload a video")
                    .font(.title2.bold())
                    .padding(.top, 12)

                Text("We'll upload the original file, no iCloud sync required.")
                    .font(.footnote)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)

                Spacer().frame(height: 12)

                Button {
                    showingRecord = true
                } label: {
                    VStack(spacing: 6) {
                        Image(systemName: "video.fill")
                            .font(.system(size: 28))
                        Text("Record new")
                            .font(.headline)
                        Text("Capture and upload immediately")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 96)
                    .padding(12)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(12)
                }
                .buttonStyle(.plain)
                .fullScreenCover(isPresented: $showingRecord) {
                    RecordView(userHash: userHash, isPresented: $showingRecord)
                        .onDisappear { isPresented = false }
                }

                Button {
                    showingPicker = true
                } label: {
                    VStack(spacing: 6) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 28))
                        Text("Choose existing")
                            .font(.headline)
                        Text("Pick a video from Photos")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 96)
                    .padding(12)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(12)
                }
                .buttonStyle(.plain)

                Spacer()
            }
            .padding(.horizontal, 16)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") { isPresented = false }
                }
            }
            .fullScreenCover(isPresented: $showingPicker) {
                PickerView(
                    userHash: userHash,
                    selectionLimit: 0,  // 0 = unlimited (Apple-defined sentinel)
                    onPicked: { url, filename in
                        // Each picked video enqueues independently as
                        // its file representation resolves. The picker
                        // dismisses itself; we dismiss this composer
                        // once we've enqueued at least one (the dialog
                        // doesn't need to stay open while uploads run).
                        UploadManager.shared.enqueue(
                            localFileURL: url,
                            assetId: "\(userHash)_\(UUID().uuidString)",
                            filename: filename,
                            userHash: userHash
                        )
                        showingPicker = false
                        isPresented = false
                    },
                    onCancelled: {
                        showingPicker = false
                    },
                    onError: { _ in
                        showingPicker = false
                    }
                )
            }
        }
    }
}
