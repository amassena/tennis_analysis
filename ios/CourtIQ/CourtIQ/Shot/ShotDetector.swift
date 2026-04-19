import Foundation

enum ShotType: String {
    case serve, forehand, backhand, unknown
}

struct DetectedShot {
    let type: ShotType
    let timestamp: TimeInterval
    let confidence: Float
    let angles: JointAngles?
    let grade: String  // A-F
}

class ShotDetector: ObservableObject {
    @Published var recentShot: DetectedShot?
    @Published var serveCount = 0
    @Published var forehandCount = 0
    @Published var backhandCount = 0

    // Wrist speed history for peak detection
    private var speedHistory: [(TimeInterval, Float)] = []
    private let peakThreshold: Float = 0.15  // normalized speed threshold
    private let minShotGap: TimeInterval = 1.5
    private var lastShotTime: TimeInterval = 0

    // Pose history for shot classification
    private var poseHistory: [PoseSnapshot] = []
    private let historyWindow = 30  // ~1 second at 30fps ML rate

    // Grade ideals (from our calibrated 3D values, adapted for 2D Vision)
    private let ideals: [ShotType: (knee: Float, trunk: Float, arm: Float)] = [
        .forehand: (knee: 148, trunk: 25, arm: 137),
        .backhand: (knee: 142, trunk: 18, arm: 139),
        .serve:    (knee: 155, trunk: 21, arm: 143),
    ]

    func processPose(_ snapshot: PoseSnapshot) {
        poseHistory.append(snapshot)
        if poseHistory.count > historyWindow {
            poseHistory.removeFirst()
        }

        speedHistory.append((snapshot.timestamp, snapshot.wristSpeed))
        // Keep 3 seconds of speed history
        let cutoff = snapshot.timestamp - 3.0
        speedHistory.removeAll { $0.0 < cutoff }

        // Detect shot via wrist speed peak
        if snapshot.wristSpeed > peakThreshold,
           isPeak(),
           snapshot.timestamp - lastShotTime > minShotGap {
            let type = classifyShot(snapshot)
            let grade = gradeShot(snapshot, type: type)
            let shot = DetectedShot(
                type: type,
                timestamp: snapshot.timestamp,
                confidence: 0.8,
                angles: snapshot.angles,
                grade: grade
            )
            lastShotTime = snapshot.timestamp
            DispatchQueue.main.async { [weak self] in
                self?.recentShot = shot
                switch type {
                case .serve:    self?.serveCount += 1
                case .forehand: self?.forehandCount += 1
                case .backhand: self?.backhandCount += 1
                case .unknown:  break
                }
            }
        }
    }

    private func isPeak() -> Bool {
        guard speedHistory.count >= 3 else { return false }
        let n = speedHistory.count
        return speedHistory[n-2].1 >= speedHistory[n-3].1 &&
               speedHistory[n-2].1 >= speedHistory[n-1].1
    }

    private func classifyShot(_ snapshot: PoseSnapshot) -> ShotType {
        // Simple heuristic classification based on wrist position relative to body
        // Phase 2 will use CoreML for proper classification
        guard let wrist = snapshot.joints[.rightWrist],
              let hip = snapshot.joints[.rightHip],
              let shoulder = snapshot.joints[.rightShoulder] else {
            return .unknown
        }

        // Serve: wrist significantly above shoulder at contact
        if wrist.y < shoulder.y - 0.1 {
            return .serve
        }

        // Forehand vs backhand: wrist position relative to hip center
        let hipCenter = snapshot.joints[.root] ?? hip
        if wrist.x > hipCenter.x + 0.05 {
            return .forehand  // wrist on dominant side
        } else if wrist.x < hipCenter.x - 0.05 {
            return .backhand
        }

        return .forehand  // default
    }

    private func gradeShot(_ snapshot: PoseSnapshot, type: ShotType) -> String {
        guard let angles = snapshot.angles,
              let ideal = ideals[type] else { return "?" }

        var yellow = 0, red = 0
        let flagYellow: Float = 15, flagRed: Float = 30

        for (value, target) in [
            (angles.knee, ideal.knee),
            (angles.trunk, ideal.trunk),
            (angles.arm, ideal.arm),
        ] {
            let delta = abs(value - target)
            if delta >= flagRed { red += 1 }
            else if delta >= flagYellow { yellow += 1 }
        }

        if red >= 2 { return "F" }
        if red == 1 { return "D" }
        if yellow >= 2 { return "C" }
        if yellow == 1 { return "B" }
        return "A"
    }

    func reset() {
        serveCount = 0
        forehandCount = 0
        backhandCount = 0
        recentShot = nil
        speedHistory = []
        poseHistory = []
        lastShotTime = 0
    }
}
