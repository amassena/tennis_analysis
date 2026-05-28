import SwiftUI
import UserNotifications
import UIKit

@main
struct CourtIQApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var auth = AuthCoordinator()
    @StateObject private var nav = AppNavigation()
    private let notifDelegate = NotificationDelegate()

    init() {
        // Foreground notifications: without this delegate iOS silently
        // suppresses banners while the app is in the foreground. We
        // explicitly opt in so "your video is ready" banners surface
        // even when the user has the Upload tab open and is watching
        // the Recent section update.
        UNUserNotificationCenter.current().delegate = notifDelegate
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(auth)
                .environmentObject(nav)
                .preferredColorScheme(.dark)
                .task {
                    UploadResumer.resumeOnLaunch()
                }
                .onReceive(NotificationCenter.default.publisher(
                    for: .didTapReadyNotification
                )) { note in
                    if let vid = note.userInfo?["video_id"] as? String {
                        nav.openGallery(anchor: vid)
                    }
                }
        }
    }
}

private final class NotificationDelegate: NSObject, UNUserNotificationCenterDelegate {
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void,
    ) {
        completionHandler([.banner, .sound, .list])
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void,
    ) {
        // Forward the tap to AppNavigation so the gallery jumps to the
        // ready video. (Done via NotificationCenter because the
        // delegate doesn't have direct access to the SwiftUI env.)
        let info = response.notification.request.content.userInfo
        NotificationCenter.default.post(
            name: .didTapReadyNotification,
            object: nil,
            userInfo: info,
        )
        completionHandler()
    }
}

extension Notification.Name {
    static let didTapReadyNotification = Notification.Name("ready.notification.tapped")
}

/// App-level orientation gate. The app is portrait-only everywhere
/// EXCEPT the fullscreen video player, which flips `orientationLock`
/// to `.landscape` while active so `requestGeometryUpdate` can rotate
/// the device. The Info.plist must permit landscape (it's the hard
/// ceiling) — this delegate is what keeps every *other* screen portrait
/// despite that. FilterablePlayerView owns flipping this flag.
final class AppDelegate: NSObject, UIApplicationDelegate {
    static var orientationLock: UIInterfaceOrientationMask = .portrait

    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?,
    ) -> UIInterfaceOrientationMask {
        AppDelegate.orientationLock
    }
}
