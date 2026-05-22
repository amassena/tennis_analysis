import SwiftUI

struct SettingsView: View {
    let userHash: String
    @Binding var isPresented: Bool

    @EnvironmentObject var auth: AuthCoordinator
    @ObservedObject private var manager = UploadManager.shared

    @State private var showingDeleteConfirm = false
    @State private var showingDeleteFinal = false
    @State private var deletingInFlight = false
    @State private var deleteError: String?

    var body: some View {
        NavigationView {
            Form {
                Section("Account") {
                    LabeledContent("Signed in as", value: userHash)
                        .textSelection(.enabled)
                    Button("Sign out") {
                        auth.signOut()
                        isPresented = false
                    }
                }

                Section("Uploads") {
                    Toggle("WiFi only", isOn: $manager.wifiOnly)
                    Text("Cellular uploads can use 1–4 GB per video. Turn off only on a paid data plan you're OK with.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section("About") {
                    Link("Privacy policy", destination: URL(string: "https://tennis.playfullife.com/privacy")!)
                    LabeledContent("Version", value: appVersionString)
                    Link("Support",
                         destination: URL(string: "mailto:amassena@gmail.com?subject=Tennis%20Uploader%20Support")!)
                }

                Section("Danger zone") {
                    Button(role: .destructive) {
                        showingDeleteConfirm = true
                    } label: {
                        if deletingInFlight {
                            HStack { ProgressView(); Text("Deleting…") }
                        } else {
                            Text("Delete my account")
                        }
                    }
                    .disabled(deletingInFlight)
                    Text("Removes your user record and every video you uploaded. This cannot be undone.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                if let deleteError {
                    Section {
                        Text(deleteError)
                            .foregroundColor(.red)
                            .font(.callout)
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { isPresented = false }
                }
            }
            .confirmationDialog(
                "Delete account?",
                isPresented: $showingDeleteConfirm,
                titleVisibility: .visible
            ) {
                Button("Continue", role: .destructive) {
                    showingDeleteFinal = true
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This removes your user record and every video you've uploaded. You'll be signed out.")
            }
            .alert("Delete everything?", isPresented: $showingDeleteFinal) {
                Button("Delete forever", role: .destructive) { performDelete() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("All your uploaded videos will be permanently deleted from the server. This cannot be undone.")
            }
        }
    }

    private var appVersionString: String {
        let v = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "—"
        let b = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "—"
        return "\(v) (\(b))"
    }

    private func performDelete() {
        deletingInFlight = true
        deleteError = nil
        Task {
            let result = await auth.deleteAccount()
            deletingInFlight = false
            switch result {
            case .success:
                isPresented = false
            case .failure(let msg):
                deleteError = msg
            }
        }
    }
}
