package main

import (
	"fmt"
	"image"
	"os"
	"time"

	"gocv.io/x/gocv"
)

var emotionLabels = []string{"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"}

type EmotionExtractor struct {
	workers chan *Worker
}

func NewEmotionExtractor(threads int) (*EmotionExtractor, error) {
	workers := make(chan *Worker, threads)
	for i := 0; i < threads; i++ {
		worker, err := NewWorker()
		if err != nil {
			return nil, fmt.Errorf("cant create worker %w", err)
		}
		workers <- worker
	}

	return &EmotionExtractor{workers: workers}, nil
}

func (emotion EmotionExtractor) NextWorker() *Worker {
	return <-emotion.workers
}

func (emotion EmotionExtractor) ReleaseWorker(worker *Worker) {
	emotion.workers <- worker
}

func (emotion EmotionExtractor) Stop() {
	threads := cap(emotion.workers)
	for i := 0; i < threads; i++ {
		worker := <-emotion.workers
		worker.Stop()
	}
}

func (emotion EmotionExtractor) getLabel(index int) string {
	return emotionLabels[index]
}

type Worker struct {
	Img        gocv.Mat
	faceNet    gocv.Net
	emotionNet gocv.Net
	ratio      float64
	mean       gocv.Scalar
}

func NewWorker() (*Worker, error) {
	faceModel := "./face.caffemodel"
	faceConfig := "./face.prototxt"
	/* model := "./emotion_miniXception.caffemodel"
	config := "./emotion_miniXception.prototxt" */
	model := "./tf/saved_model.pb"
	config := ""
	/* model := "./emotion.caffemodel"
	config := "./emotion.prototxt" */
	backend := gocv.NetBackendDefault
	if len(os.Args) > 5 {
		backend = gocv.ParseNetBackend(os.Args[5])
	}

	target := gocv.NetTargetCPU
	if len(os.Args) > 6 {
		target = gocv.ParseNetTarget(os.Args[6])
	}
	// open DNN classifier
	// net := gocv.ReadNet(model, config)
	net := gocv.ReadNetFromTensorflow(model)
	if net.Empty() {
		return nil, fmt.Errorf("Reading model : %v %v\n", model, config)
	}
	net.SetPreferableBackend(gocv.NetBackendType(backend))
	net.SetPreferableTarget(gocv.NetTargetType(target))

	// open DNN object tracking model
	faceNet := gocv.ReadNet(faceModel, faceConfig)
	if faceNet.Empty() {
		return nil, fmt.Errorf("Reading model : %v %v\n", faceModel, faceConfig)
	}
	img := gocv.NewMat()
	faceNet.SetPreferableBackend(gocv.NetBackendType(backend))
	faceNet.SetPreferableTarget(gocv.NetTargetType(target))
	worker := &Worker{
		faceNet:    faceNet,
		emotionNet: net,
		Img:        img,
	}
	return worker, nil
}
func (worker *Worker) Stop() {
	worker.faceNet.Close()
	worker.emotionNet.Close()
	worker.Img.Close()
}

func (worker *Worker) Predict() (label string, maxVal float32) {
	ratio := 1.0
	mean := gocv.NewScalar(104, 117, 123, 0)
	img := worker.Img
	faceNet := worker.faceNet
	emotionNet := worker.emotionNet
	st := time.Now()
	// convert image Mat to 300x300 blob that the object detector can analyze
	blob := gocv.BlobFromImage(img, ratio, image.Pt(300, 300), mean, false, false)
	defer blob.Close()

	// feed the blob into the detector
	faceNet.SetInput(blob, "")

	// run a forward pass thru the network
	prob := faceNet.Forward("")
	defer prob.Close()

	rects := performDetection(&img, prob)
	fmt.Println("face step", time.Since(st))

	for _, r := range rects {
		st := time.Now()
		roi := img.Region(r)
		roi_gray := gocv.NewMat()
		gocv.CvtColor(roi, &roi_gray, gocv.ColorRGBToGray)
		gocv.IMWrite("current_roi.jpg", roi)
		blob := gocv.BlobFromImage(roi_gray, ratio, image.Pt(64, 64), mean, false, false)
		// blob := gocv.BlobFromImage(roi, ratio, image.Pt(224, 224), mean, false, false)
		defer blob.Close()

		emotionNet.SetInput(blob, "")

		prob := emotionNet.Forward("")
		defer prob.Close()

		probMat := prob.Reshape(1, 1)
		defer probMat.Close()

		// determine the most probable classification
		_, maxVal, _, maxLoc := gocv.MinMaxLoc(probMat)
		fmt.Println(maxLoc, maxVal)

		fmt.Println("emotion step", time.Since(st))
		return emotionLabels[maxLoc.X], maxVal
	}
	return
}

// performDetection analyzes the results from the detector network,
// which produces an output blob with a shape 1x1xNx7
// where N is the number of detections, and each detection
// is a vector of float values
// [batchId, classId, confidence, left, top, right, bottom]
func performDetection(frame *gocv.Mat, results gocv.Mat) (rects []image.Rectangle) {
	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		if confidence > 0.5 {
			left := int(results.GetFloatAt(0, i+3) * float32(frame.Cols()))
			top := int(results.GetFloatAt(0, i+4) * float32(frame.Rows()))
			right := int(results.GetFloatAt(0, i+5) * float32(frame.Cols()))
			bottom := int(results.GetFloatAt(0, i+6) * float32(frame.Rows()))
			rect := image.Rect(left, top, right, bottom)
			rects = append(rects, rect)
		}
	}
	return rects
}
