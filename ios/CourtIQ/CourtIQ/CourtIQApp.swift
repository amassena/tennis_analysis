import SwiftUI

@main
struct CourtIQApp: App {
    @StateObject private var auth = AuthCoordinator()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(auth)
                .preferredColorScheme(.dark)
        }
    }
}
