import org.opencv.core.Core;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;

public class Solution {

	public static void main(String[] args) {

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String temp_addr = "F:/JAVA/openCV/InterestPoint1/images/IMG_20161001_233351.jpg";
		String main_addr = "F:/JAVA/openCV/InterestPoint1/images/IMG_20161001_233436.jpg";
		Mat temp = Imgcodecs.imread(temp_addr,Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR);
		Mat obj = Imgcodecs.imread(main_addr, Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR);
		MatOfKeyPoint tempKeyPoints = new MatOfKeyPoint();
		FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
		featureDetector.detect(temp, tempKeyPoints);
		KeyPoint[] keyPoint = tempKeyPoints.toArray();
		System.out.println(keyPoint[0]);
		
		MatOfKeyPoint tempDescriptor = new MatOfKeyPoint();
		DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		descriptorExtractor.compute(temp, tempKeyPoints, tempDescriptor);
		
		//creating another image to preserve original template
		Mat temp_bc = new Mat(temp.rows(),temp.cols(),Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR);
		Scalar newKeyPointColor = new Scalar(255,0,0);
		Features2d.drawKeypoints(temp, tempKeyPoints, temp_bc, newKeyPointColor,0);
		Imgcodecs.imwrite("F:/JAVA/openCV/InterestPoint1/images/IMG_20161001_233351(processed).jpg", temp_bc);
		
		
	}

}
