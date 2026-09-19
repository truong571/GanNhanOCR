import Cocoa
import Vision

let args = CommandLine.arguments
if args.count < 2 {
    print("Usage: ocr_quocngu <image_path>")
    exit(1)
}

let imagePath = args[1]
guard let image = NSImage(contentsOfFile: imagePath),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    fputs("Cannot load image: \(imagePath)\n", stderr)
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLanguages = ["vi-VN", "fr-FR", "en-US"]
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
    guard let observations = request.results else { exit(0) }
    
    // Sort observations top-to-bottom, left-to-right
    // Vision coordinates: origin is lower-left (0,0), boundingBox is (x, y, w, h)
    let sortedObs = observations.sorted { (a, b) -> Bool in
        // Compare y top-to-bottom (higher y is higher on the page)
        if abs(a.boundingBox.origin.y - b.boundingBox.origin.y) > 0.015 {
            return a.boundingBox.origin.y > b.boundingBox.origin.y
        }
        return a.boundingBox.origin.x < b.boundingBox.origin.x
    }
    
    for obs in sortedObs {
        if let candidate = obs.topCandidates(1).first {
            let bbox = obs.boundingBox
            // Print: y_top x_left text
            let yTop = 1.0 - (bbox.origin.y + bbox.size.height)
            print(String(format: "%.3f\t%.3f\t%@", yTop, bbox.origin.x, candidate.string))
        }
    }
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
