import SwiftUI
import AVKit

struct SessionReviewView: View {
    @ObservedObject var recording: SessionRecording
    @Binding var isPresented: Bool
    @State private var selectedShot: RecordedShot?
    @State private var showingPlayer = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 0) {
                    // Session summary header
                    SessionSummaryCard(recording: recording)
                        .padding()

                    // Shot list
                    LazyVStack(spacing: 12) {
                        ForEach(recording.shots) { shot in
                            ShotCard(shot: shot) {
                                selectedShot = shot
                                showingPlayer = true
                            }
                        }
                    }
                    .padding(.horizontal)

                    if recording.shots.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "figure.tennis")
                                .font(.system(size: 48))
                                .foregroundColor(.gray)
                            Text("No shots detected")
                                .font(.headline)
                                .foregroundColor(.gray)
                            Text("Try recording a session with more swings")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 60)
                    }
                }
            }
            .background(Color.black)
            .navigationTitle("Session Review")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") { isPresented = false }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: uploadSession) {
                        if recording.isUploading {
                            ProgressView()
                                .tint(.white)
                        } else {
                            Label("Upload", systemImage: "icloud.and.arrow.up")
                        }
                    }
                    .disabled(recording.isUploading)
                }
            }
            .sheet(isPresented: $showingPlayer) {
                if let shot = selectedShot, let url = recording.videoURL {
                    ShotReplayView(shot: shot, videoURL: url)
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    func uploadSession() {
        // TODO: Direct R2 upload
        recording.isUploading = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            recording.isUploading = false
        }
    }
}

// MARK: - Session Summary
struct SessionSummaryCard: View {
    @ObservedObject var recording: SessionRecording

    var gradeDistribution: [(String, Int)] {
        var counts: [String: Int] = [:]
        for shot in recording.shots {
            counts[shot.grade, default: 0] += 1
        }
        return ["A", "B", "C", "D", "F"].compactMap { g in
            counts[g].map { (g, $0) }
        }
    }

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("\(recording.shots.count) shots")
                        .font(.system(size: 28, weight: .bold))
                        .foregroundColor(.white)
                    Text(formatDuration(recording.duration))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()

                // Shot type breakdown
                HStack(spacing: 14) {
                    typePill("S", count: recording.shots.filter { $0.type == .serve }.count, color: .orange)
                    typePill("FH", count: recording.shots.filter { $0.type == .forehand }.count, color: .green)
                    typePill("BH", count: recording.shots.filter { $0.type == .backhand }.count, color: .blue)
                }
            }

            // Grade distribution bar
            if !gradeDistribution.isEmpty {
                HStack(spacing: 2) {
                    ForEach(gradeDistribution, id: \.0) { grade, count in
                        let pct = Double(count) / max(1, Double(recording.shots.count))
                        Rectangle()
                            .fill(gradeColor(grade))
                            .frame(height: 6)
                            .frame(maxWidth: .infinity)
                            .scaleEffect(x: pct > 0 ? 1 : 0, anchor: .leading)
                    }
                }
                .cornerRadius(3)

                HStack(spacing: 8) {
                    ForEach(gradeDistribution, id: \.0) { grade, count in
                        HStack(spacing: 3) {
                            Circle()
                                .fill(gradeColor(grade))
                                .frame(width: 8, height: 8)
                            Text("\(grade): \(count)")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }
                    }
                    Spacer()
                }
            }
        }
        .padding()
        .background(Color(.systemGray6).opacity(0.15))
        .cornerRadius(12)
    }

    func typePill(_ label: String, count: Int, color: Color) -> some View {
        VStack(spacing: 2) {
            Text("\(count)")
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.white)
            Text(label)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(color)
        }
    }

    func gradeColor(_ grade: String) -> Color {
        switch grade {
        case "A": return .green
        case "B": return Color(red: 0.47, green: 0.87, blue: 0.47)
        case "C": return .yellow
        case "D": return .orange
        case "F": return .red
        default: return .gray
        }
    }

    func formatDuration(_ seconds: TimeInterval) -> String {
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return "\(m)m \(s)s"
    }
}

// MARK: - Shot Card with inline filmstrip
struct ShotCard: View {
    let shot: RecordedShot
    let onTap: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header: type + grade + timestamp
            HStack {
                Text(shot.type.rawValue.uppercased())
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(shotColor)

                Text(shot.grade)
                    .font(.system(size: 14, weight: .black))
                    .foregroundColor(.black)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(gradeColor)
                    .cornerRadius(4)

                if let angles = shot.angles {
                    Text("K:\(Int(angles.knee))° T:\(Int(angles.trunk))° A:\(Int(angles.arm))°")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.secondary)
                }

                Spacer()

                Text(formatTimestamp(shot.timestamp))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            // Filmstrip (if frames captured)
            if !shot.frames.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 2) {
                        ForEach(0..<shot.frames.count, id: \.self) { i in
                            Image(uiImage: shot.frames[i])
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 80, height: 120)
                                .clipped()
                                .cornerRadius(4)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 4)
                                        .stroke(i == shot.frames.count / 2 ? Color.orange : Color.clear, lineWidth: 2)
                                )
                        }
                    }
                }
                .frame(height: 120)
            }

            // Tap to replay
            Button(action: onTap) {
                HStack {
                    Image(systemName: "play.fill")
                        .font(.system(size: 10))
                    Text("Replay in slow-mo")
                        .font(.system(size: 12, weight: .medium))
                }
                .foregroundColor(.orange)
            }
        }
        .padding(12)
        .background(Color(.systemGray6).opacity(0.1))
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color(.systemGray4).opacity(0.3), lineWidth: 1)
        )
    }

    var shotColor: Color {
        switch shot.type {
        case .serve: return .orange
        case .forehand: return .green
        case .backhand: return .blue
        case .unknown: return .gray
        }
    }

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

    func formatTimestamp(_ t: TimeInterval) -> String {
        let m = Int(t) / 60
        let s = Int(t) % 60
        return String(format: "%d:%02d", m, s)
    }
}

// MARK: - Shot Replay (slow-mo at the shot's timestamp)
struct ShotReplayView: View {
    let shot: RecordedShot
    let videoURL: URL
    @State private var player: AVPlayer?

    var body: some View {
        VStack {
            if let player = player {
                VideoPlayer(player: player)
                    .onAppear {
                        let time = CMTime(seconds: max(0, shot.timestamp - 0.5), preferredTimescale: 600)
                        player.seek(to: time)
                        player.rate = 0.25 // slow-mo
                    }
            } else {
                ProgressView()
            }

            Text("\(shot.type.rawValue.uppercased()) #\(shot.index + 1) — \(shot.grade)")
                .font(.headline)
                .foregroundColor(.white)
                .padding()
        }
        .background(.black)
        .onAppear {
            player = AVPlayer(url: videoURL)
        }
    }
}
