import SwiftUI
import Vision

struct PoseOverlayView: View {
    @ObservedObject var session: TennisSession

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Skeleton lines
                if let pose = session.currentPose {
                    SkeletonShape(pose: pose, size: geo.size)
                        .stroke(Color.green.opacity(0.7), lineWidth: 2.5)
                }

                // Shot flash overlay
                if let shot = session.shotDetector.recentShot,
                   Date().timeIntervalSince1970 - shot.timestamp < 2.0 {
                    ShotBadgeView(shot: shot)
                        .position(x: geo.size.width / 2, y: 120)
                        .transition(.scale.combined(with: .opacity))
                }
            }
        }
    }
}

struct ShotBadgeView: View {
    let shot: DetectedShot

    var gradeColor: Color {
        switch shot.grade {
        case "A": return .green
        case "B": return Color(red: 0.47, green: 0.87, blue: 0.47)
        case "C": return .yellow
        case "D": return .orange
        case "F": return .red
        default: return .gray
        }
    }

    var shotColor: Color {
        switch shot.type {
        case .serve: return .orange
        case .forehand: return .green
        case .backhand: return .blue
        case .unknown: return .gray
        }
    }

    var body: some View {
        HStack(spacing: 12) {
            Text(shot.type.rawValue.uppercased())
                .font(.system(size: 22, weight: .bold))
                .foregroundColor(shotColor)

            Text(shot.grade)
                .font(.system(size: 24, weight: .black))
                .foregroundColor(.black)
                .frame(width: 40, height: 40)
                .background(gradeColor)
                .cornerRadius(8)

            if let angles = shot.angles {
                VStack(alignment: .leading, spacing: 2) {
                    angleLabel("Knee", value: angles.knee)
                    angleLabel("Trunk", value: angles.trunk)
                    angleLabel("R.Arm", value: angles.arm)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.black.opacity(0.75))
        .cornerRadius(12)
    }

    func angleLabel(_ name: String, value: Float) -> some View {
        Text("\(name) \(Int(value))°")
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundColor(.white.opacity(0.9))
    }
}

struct SkeletonShape: Shape {
    let pose: PoseSnapshot
    let size: CGSize

    // Skeleton connections (pairs of joints to draw lines between)
    static let connections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
        // Torso
        (.leftShoulder, .rightShoulder),
        (.leftShoulder, .leftHip),
        (.rightShoulder, .rightHip),
        (.leftHip, .rightHip),
        // Left arm
        (.leftShoulder, .leftElbow),
        (.leftElbow, .leftWrist),
        // Right arm
        (.rightShoulder, .rightElbow),
        (.rightElbow, .rightWrist),
        // Left leg
        (.leftHip, .leftKnee),
        (.leftKnee, .leftAnkle),
        // Right leg
        (.rightHip, .rightKnee),
        (.rightKnee, .rightAnkle),
        // Neck
        (.neck, .leftShoulder),
        (.neck, .rightShoulder),
    ]

    func path(in rect: CGRect) -> Path {
        var path = Path()

        for (from, to) in Self.connections {
            guard let p1 = pose.joints[from],
                  let p2 = pose.joints[to] else { continue }

            let pt1 = CGPoint(x: p1.x * size.width, y: p1.y * size.height)
            let pt2 = CGPoint(x: p2.x * size.width, y: p2.y * size.height)

            path.move(to: pt1)
            path.addLine(to: pt2)
        }

        // Joint dots
        for (_, point) in pose.joints {
            let pt = CGPoint(x: point.x * size.width, y: point.y * size.height)
            path.addEllipse(in: CGRect(x: pt.x - 4, y: pt.y - 4, width: 8, height: 8))
        }

        return path
    }
}
