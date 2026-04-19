import Vision
import simd

struct JointAngles {
    let knee: Float     // hip-knee-ankle angle
    let trunk: Float    // shoulder-line vs hip-line rotation
    let arm: Float      // shoulder-elbow-wrist angle
}

struct PoseSnapshot {
    let joints: [VNHumanBodyPoseObservation.JointName: CGPoint]
    let confidence: [VNHumanBodyPoseObservation.JointName: Float]
    let timestamp: TimeInterval
    let angles: JointAngles?
    let wristSpeed: Float  // pixels per second (for shot detection)
}

class PoseProcessor {

    private var lastWristPosition: CGPoint?
    private var lastTimestamp: TimeInterval = 0

    // Apple Vision joint name mapping to our analysis joints
    // Right side = racket arm for right-handed player
    static let analysisJoints: [VNHumanBodyPoseObservation.JointName] = [
        .rightShoulder, .rightElbow, .rightWrist,
        .leftShoulder, .leftElbow, .leftWrist,
        .rightHip, .rightKnee, .rightAnkle,
        .leftHip, .leftKnee, .leftAnkle,
        .root, .neck,
    ]

    func process(_ observation: VNHumanBodyPoseObservation, at timestamp: TimeInterval) -> PoseSnapshot? {
        guard let points = try? observation.recognizedPoints(.all) else { return nil }

        var joints: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:]
        var confs: [VNHumanBodyPoseObservation.JointName: Float] = [:]

        for (name, point) in points where point.confidence > 0.3 {
            // Vision coordinates: origin bottom-left, normalized 0-1
            joints[name] = CGPoint(x: point.location.x, y: 1.0 - point.location.y)
            confs[name] = point.confidence
        }

        // Wrist speed (right wrist, pixels/sec — used for shot detection)
        var wristSpeed: Float = 0
        if let wrist = joints[.rightWrist] {
            if let last = lastWristPosition, timestamp > lastTimestamp {
                let dt = Float(timestamp - lastTimestamp)
                let dx = Float(wrist.x - last.x)
                let dy = Float(wrist.y - last.y)
                wristSpeed = sqrt(dx * dx + dy * dy) / dt
            }
            lastWristPosition = wrist
            lastTimestamp = timestamp
        }

        let angles = computeAngles(joints)

        return PoseSnapshot(
            joints: joints,
            confidence: confs,
            timestamp: timestamp,
            angles: angles,
            wristSpeed: wristSpeed
        )
    }

    private func computeAngles(_ joints: [VNHumanBodyPoseObservation.JointName: CGPoint]) -> JointAngles? {
        // Knee: right hip → right knee → right ankle
        guard let rHip = joints[.rightHip],
              let rKnee = joints[.rightKnee],
              let rAnkle = joints[.rightAnkle],
              let rShoulder = joints[.rightShoulder],
              let rElbow = joints[.rightElbow],
              let rWrist = joints[.rightWrist],
              let lShoulder = joints[.leftShoulder],
              let lHip = joints[.leftHip] else {
            return nil
        }

        let knee = angleBetween(a: rHip, b: rKnee, c: rAnkle)
        let arm = angleBetween(a: rShoulder, b: rElbow, c: rWrist)

        // Trunk rotation: angle between shoulder-line and hip-line
        let shVec = CGPoint(x: rShoulder.x - lShoulder.x, y: rShoulder.y - lShoulder.y)
        let hpVec = CGPoint(x: rHip.x - lHip.x, y: rHip.y - lHip.y)
        let aSh = atan2(shVec.y, shVec.x)
        let aHp = atan2(hpVec.y, hpVec.x)
        var trunk = abs(aSh - aHp)
        if trunk > .pi { trunk = 2 * .pi - trunk }
        trunk = trunk * 180 / .pi

        return JointAngles(knee: knee, trunk: Float(trunk), arm: arm)
    }

    private func angleBetween(a: CGPoint, b: CGPoint, c: CGPoint) -> Float {
        let ba = SIMD2<Float>(Float(a.x - b.x), Float(a.y - b.y))
        let bc = SIMD2<Float>(Float(c.x - b.x), Float(c.y - b.y))
        let dot = simd_dot(ba, bc)
        let mag = simd_length(ba) * simd_length(bc)
        guard mag > 1e-6 else { return 0 }
        let cos = max(-1.0, min(1.0, dot / mag))
        return acos(cos) * 180 / .pi
    }
}
