import java.io.IOException;
import java.lang.reflect.Field;
import java.util.LinkedList;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;

public class Solution {
	
	

	public static void main(String[] args) throws IOException, IllegalArgumentException, IllegalAccessException, NoSuchFieldException, SecurityException{

		System.setProperty("java.library.path", "G:\\installed\\openCV_2.4\\opencv\\build\\java\\x64");

		Field fieldSysPath = ClassLoader.class.getDeclaredField( "sys_paths" );
	    fieldSysPath.setAccessible( true );
	    fieldSysPath.set( null, null );
	    
	    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String temp_addr = args[0];
		String obj_addr = args[1];
		
		//Processing template part
		Mat temp = Highgui.imread(temp_addr);
		MatOfKeyPoint tempKeyPoints = new MatOfKeyPoint();
		FeatureDetector keyPointTemp = FeatureDetector.create(FeatureDetector.SURF);
		keyPointTemp.detect(temp, tempKeyPoints);
		MatOfKeyPoint tempDescriptorKeyPoint = new MatOfKeyPoint();
		DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		descriptorExtractor.compute(temp, tempKeyPoints, tempDescriptorKeyPoint);
		
		//Processing real time image part
		Mat obj = Highgui.imread(obj_addr);
		MatOfKeyPoint objKeyPoint = new MatOfKeyPoint();
		FeatureDetector keyPointObj = FeatureDetector.create(FeatureDetector.SURF);
		keyPointObj.detect(obj, objKeyPoint);
		MatOfKeyPoint objDescriptorKeyPoint = new MatOfKeyPoint();
		DescriptorExtractor descriptorObj = DescriptorExtractor.create(DescriptorExtractor.SURF);
		descriptorObj.compute(obj, objKeyPoint, objDescriptorKeyPoint);
		
		//matching the descriptor for identifying identical points in images
		LinkedList<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
		LinkedList<DMatch> goodMatch = new LinkedList<DMatch>();
		DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
		descriptorMatcher.knnMatch(tempDescriptorKeyPoint, objDescriptorKeyPoint, matches, 2);
		
		//isolating best matching points in a separate list
		float nndrRatio = 0.75f;
		MatOfDMatch matOfDMatches;
		DMatch[] dMatchArr;
		DMatch mat1,mat2;
		
		for(int i = 0 ; i < matches.size() ; i++){
			matOfDMatches = matches.get(i);
			dMatchArr = matOfDMatches.toArray();
			mat1 = dMatchArr[0];
			mat2 = dMatchArr[1];
			
			if(mat1.distance <= mat2.distance*nndrRatio){
				goodMatch.addLast(mat1);
			}
		}
		
		if(goodMatch.size() > 500){
			System.out.println("a match is found");
		}
		else{
			System.out.println("No match found");
		}
	}
}