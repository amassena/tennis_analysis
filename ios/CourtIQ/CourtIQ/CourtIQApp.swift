import SwiftUI

@main
struct CourtIQApp: App {
    @StateObject private var auth = AuthCoordinator()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(auth)
                .preferredColorScheme(.dark)
                .task {
                    // Pick up any in-flight uploads from a prior session.
                    UploadResumer.resumeOnLaunch()
                }
        }
    }
}
