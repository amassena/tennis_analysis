import Foundation
import Combine
import SwiftUI
import UserNotifications

/// One row in `GET /api/u/<hash>/recent`.
struct RecentUpload: Codable, Identifiable, Equatable {
    let video_id: String
    let filename: String?
    let status: String
    let stage: String?
    let progress: Int?
    let uploaded_at: String?
    let updated_at: String?
    let error: String?
    let gallery_url: String?

    var id: String { video_id }

    var isComplete: Bool { status == "complete" }
    var isFailed: Bool { status == "failed" }
    var isInflight: Bool { !isComplete && !isFailed }
}

private struct RecentResponse: Codable {
    let user_hash: String
    let count: Int
    let items: [RecentUpload]
}

/// Drives the "Recently uploaded" section in the Upload tab. Polls the
/// worker every `pollInterval` seconds while the view is foregrounded.
@MainActor
final class RecentUploadsModel: ObservableObject {
    @Published private(set) var items: [RecentUpload] = []
    @Published private(set) var lastError: String?
    @Published private(set) var isLoading = false
    /// True once a refresh has SUCCEEDED at least once (even if it returned
    /// zero items). Until then `items.isEmpty` just means "haven't loaded
    /// yet" — the Upload tab must not show the "Upload your first video"
    /// screen during that window, or it flashes on every launch for users
    /// who actually have videos.
    @Published private(set) var hasLoadedOnce = false

    let userHash: String
    private var timer: AnyCancellable?
    private let pollInterval: TimeInterval = 15

    init(userHash: String) {
        self.userHash = userHash
        // Show the last-known list instantly on launch so there's no
        // full-screen spinner / loading flicker while the (2–3s) fetch runs,
        // and so a slow/failed fetch still shows the user's videos. Treat a
        // cache hit as "loaded once" — the background refresh updates it.
        if let cached = Self.loadCache(userHash: userHash), !cached.isEmpty {
            items = cached
            hasLoadedOnce = true
        }
    }

    private static func cacheKey(_ h: String) -> String { "recentCache_\(h)" }
    private static func loadCache(userHash: String) -> [RecentUpload]? {
        guard let data = UserDefaults.standard.data(forKey: cacheKey(userHash)) else { return nil }
        return try? JSONDecoder().decode([RecentUpload].self, from: data)
    }
    private func saveCache() {
        guard let data = try? JSONEncoder().encode(items) else { return }
        UserDefaults.standard.set(data, forKey: Self.cacheKey(userHash))
    }

    func start() {
        Task { await self.refresh() }
        timer = Timer.publish(every: pollInterval, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                Task { await self?.refresh() }
            }
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let resp: RecentResponse = try await APIClient.get(
                path: "api/u/\(userHash)/recent?limit=25",
            )
            let previous = items
            if resp.items != items { items = resp.items }
            lastError = nil
            hasLoadedOnce = true
            saveCache()
            // Fire a local notification for every video that just
            // transitioned to "complete" since the last poll. iOS
            // delivers banners even when the app is foregrounded if we
            // opt in via UNUserNotificationCenterDelegate (set up in
            // CourtIQApp).
            notifyReadyTransitions(from: previous, to: resp.items)
        } catch APIClient.APIError.unauthorized {
            lastError = "Session expired"
        } catch {
            lastError = error.localizedDescription
        }
    }

    private func notifyReadyTransitions(from prev: [RecentUpload], to next: [RecentUpload]) {
        // First-ever load: don't fire notifications for items that were
        // already complete before the app launched.
        if prev.isEmpty && !firedFirstLoad {
            firedFirstLoad = true
            return
        }
        let prevByID = Dictionary(uniqueKeysWithValues: prev.map { ($0.video_id, $0) })
        for item in next {
            guard item.isComplete else { continue }
            let wasComplete = prevByID[item.video_id]?.isComplete ?? false
            if wasComplete { continue }
            scheduleReadyNotification(for: item)
        }
    }
    private var firedFirstLoad = false

    private func scheduleReadyNotification(for item: RecentUpload) {
        UNUserNotificationCenter.current().getNotificationSettings { settings in
            switch settings.authorizationStatus {
            case .authorized, .provisional:
                Self.fire(item: item)
            case .notDetermined:
                UNUserNotificationCenter.current().requestAuthorization(
                    options: [.alert, .sound]
                ) { granted, _ in
                    if granted { Self.fire(item: item) }
                }
            default:
                break  // denied/ephemeral — respect the user's choice
            }
        }
    }

    nonisolated private static func fire(item: RecentUpload) {
        let content = UNMutableNotificationContent()
        content.title = "Session ready"
        content.body = "\(item.filename ?? item.video_id) finished processing — tap to view."
        content.sound = .default
        content.userInfo = ["video_id": item.video_id]
        let req = UNNotificationRequest(
            identifier: "ready.\(item.video_id)",
            content: content,
            trigger: nil,
        )
        UNUserNotificationCenter.current().add(req)
    }
}
