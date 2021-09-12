package main

import (
	"fmt"
	"time"

	"gocv.io/x/gocv"
)

func main() {
	// parse args
	deviceID := "0"

	// open capture device
	// file := "./test.mp4"
	// webcam, err := gocv.VideoCaptureFile(file)
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	window := gocv.NewWindow("Caffe Classifier")
	defer window.Close()

	/* status := "Ready"
	statusColor := color.RGBA{0, 0, 255, 0} */
	fmt.Printf("Start reading device: %v\n", deviceID)
	count := 0
	countEmotions := map[string]uint64{
		"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surprise": 0,
	}
	timeStart := time.Now()
	defer func() {
		fmt.Println("Frames:", count)
		fmt.Printf("Count Emotions\n%+v\n", countEmotions)
		fmt.Println("Elapsed:", time.Since(timeStart))
	}()

	extractor, err := NewEmotionExtractor(4)
	if err != nil {
		panic(err)
	}

	// startProgressInfo(webcam)

	img := gocv.NewMat()
	defer img.Close()
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		worker := extractor.NextWorker()
		img.CopyTo(&worker.Img)
		go func() {
			defer extractor.ReleaseWorker(worker)
			st := time.Now()
			label, maxVal := worker.Predict()
			fmt.Println(count, "steps", time.Since(st), label)
			count++
			if maxVal == 0 {
				return
			}
			// status := fmt.Sprintf("description: %v, maxVal: %v\n", label, maxVal)
			// fmt.Println(status)
			countEmotions[label]++
		}()
		// gocv.PutText(&img, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)

		window.IMShow(img)
		key := window.WaitKey(1e4)
		if key == 27 {
			break
		}
		webcam.Grab(15)
	}
}

func startProgressInfo(stream *gocv.VideoCapture) {
	fps := stream.Get(gocv.VideoCaptureFPS)
	totalMs := int64(stream.Get(gocv.VideoCaptureFrameCount)/fps) * 1000
	fmt.Printf("FPS: %f, Time: %s\n", fps, time.Duration(totalMs)*time.Millisecond)

	timeStart := time.Now()
	go func() {
		for {
			time.Sleep(time.Second * 5)
			if !stream.IsOpened() {
				return
			}
			currentMs := int64(stream.Get(gocv.VideoCapturePosMsec))
			progress := currentMs * 100 / totalMs
			if progress == 0 {
				progress = 1
			}
			left := time.Duration((int64(time.Since(timeStart)) / progress) * (100 - progress))
			fmt.Printf("\rProgress: %d%% %s", progress, left)
		}
	}()
}
