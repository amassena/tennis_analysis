import SwiftUI

@main
struct CourtIQApp: App {
    @StateObject private var auth = AuthCoordinator()
    @StateObject private var nav = AppNavigation()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(auth)
                .environmentObject(nav)
                .preferredColorScheme(.dark)
                .task {
                    // Pick up any in-flight uploads from a prior session.
                    UploadResumer.resumeOnLaunch()
                }
        }
    }
}
