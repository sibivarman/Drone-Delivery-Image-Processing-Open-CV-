package image_Object_Serializer;

import java.io.IOException;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;

public class Solution{

	public static void main(String[] args) throws IOException{

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String temp_addr = "E:\\images\\cofee.jpg";
		String obj_addr = "";
		
		//Processing template part
		Mat temp = Highgui.imread(temp_addr);
		MatOfKeyPoint tempKeyPoints = new MatOfKeyPoint();
		FeatureDetector keyPointDetector = FeatureDetector.create(FeatureDetector.SURF);
		keyPointDetector.detect(temp, tempKeyPoints);
		MatOfKeyPoint descriptorKeyPoint = new MatOfKeyPoint();
		DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		descriptorExtractor.compute(temp, tempKeyPoints, descriptorKeyPoint);
		
		//Processing real time image part
		Mat obj = Highgui.imread(obj_addr);
		MatOfKeyPoint objKeyPoint = new MatOfKeyPoint();
	}

}
