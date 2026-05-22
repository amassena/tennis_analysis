import SwiftUI

/// The "+ → Record or Pick" choice screen shown from the Upload tab.
struct UploadComposerSheet: View {
    let userHash: String
    @Binding var isPresented: Bool

    @State private var showingPicker = false
    // Record flow lands in PR 3; gate the button so PR 2 still ships clean.
    @State private var showingRecordUnavailable = false

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
                    showingRecordUnavailable = true
                } label: {
                    VStack(spacing: 6) {
                        Image(systemName: "video.fill")
                            .font(.system(size: 28))
                        Text("Record new")
                            .font(.headline)
                        Text("Coming in next update")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 96)
                    .padding(12)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(12)
                }
                .buttonStyle(.plain)
                .alert("Coming soon", isPresented: $showingRecordUnavailable) {
                    Button("OK", role: .cancel) {}
                } message: {
                    Text("In-app recording lands in the next update. For now, choose an existing video.")
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
                    onPicked: { url, filename in
                        showingPicker = false
                        UploadManager.shared.enqueue(
                            localFileURL: url,
                            assetId: "\(userHash)_\(UUID().uuidString)",
                            filename: filename,
                            userHash: userHash
                        )
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
