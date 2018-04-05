import java.util.List;
import java.util.LinkedList;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;

public class Solution {

	public static void main(String[] args) {

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String temp_addr = "F:/JAVA/openCV/InterestPoint1/images/img (1).jpg";
		String main_addr = "F:/JAVA/openCV/InterestPoint1/images/img (10).jpg";
		Mat temp = Highgui.imread(temp_addr);
		Mat obj = Highgui.imread(main_addr);
		MatOfKeyPoint tempKeyPoints = new MatOfKeyPoint();
		FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
		featureDetector.detect(temp, tempKeyPoints);
		MatOfKeyPoint tempDescriptor = new MatOfKeyPoint();
		DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		descriptorExtractor.compute(temp, tempKeyPoints, tempDescriptor);
		
		//creating another image to preserve original template
		Mat temp_bc = new Mat(temp.rows(),temp.cols(),Highgui.CV_LOAD_IMAGE_ANYCOLOR);
		Scalar newKeyPointColor = new Scalar(255,0,0);
		Features2d.drawKeypoints(temp, tempKeyPoints, temp_bc, newKeyPointColor,0);
		Highgui.imwrite("F:/JAVA/openCV/InterestPoint1/images/temp_processed.jpg", temp_bc);
		
		MatOfKeyPoint objKeyPoints = new MatOfKeyPoint();
		FeatureDetector featuredetector = FeatureDetector.create(FeatureDetector.SURF);
		featuredetector.detect(obj, objKeyPoints);
		
		MatOfKeyPoint objDescriptor = new MatOfKeyPoint();
		DescriptorExtractor descriptorextractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		descriptorextractor.compute(obj, objKeyPoints, objDescriptor);
		
		Mat Obj_bc = new Mat(obj.rows(),obj.cols(),Highgui.CV_LOAD_IMAGE_ANYCOLOR);
		Scalar newObjKeyPointColor = new Scalar(255,0,0);
		Features2d.drawKeypoints(obj, objKeyPoints, Obj_bc, newObjKeyPointColor, 0);
		Highgui.imwrite("F:/JAVA/openCV/InterestPoint1/images/obj_processed.jpg", Obj_bc);
		
		LinkedList<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
		LinkedList<DMatch> goodMatch = new LinkedList<DMatch>();
		DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
		descriptorMatcher.knnMatch(tempDescriptor, objDescriptor, matches, 2);
		
		float nndrRatio = 0.75f;
		
		for(int i = 0 ; i < matches.size() ; i++){
			MatOfDMatch  matOfDMatches = matches.get(i);
			DMatch[] dMatchArr = matOfDMatches.toArray();
			DMatch mat1 = dMatchArr[0];
			DMatch mat2 = dMatchArr[1];
			
			if(mat1.distance <= mat2.distance*nndrRatio){
				goodMatch.addLast(mat1);
			}
		}
		
		if(goodMatch.size() >= 7){
			System.out.println("match found");
			List<KeyPoint> tempKeyPoint =  tempKeyPoints.toList();
			List<KeyPoint> objKeyPoint = objKeyPoints.toList();
			
			LinkedList<Point> tempPoint = new LinkedList<>();
			LinkedList<Point> objPoint = new LinkedList<>();
			
			for(int i = 0 ; i < goodMatch.size() ; i++){
				tempPoint.addLast(tempKeyPoint.get(goodMatch.get(i).queryIdx).pt);
				objPoint.addLast(objKeyPoint.get(goodMatch.get(i).trainIdx).pt);
			}
			
			MatOfPoint2f tempMatOfPoint2f = new MatOfPoint2f();
			tempMatOfPoint2f.fromList(tempPoint);
			MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
			objMatOfPoint2f.fromList(objPoint);
			
			Mat homography = Calib3d.findHomography(tempMatOfPoint2f, objMatOfPoint2f, Calib3d.RANSAC, 3);
			Mat temp_corners = new Mat(4,1,CvType.CV_32FC2);
			Mat obj_corners = new Mat(4,1,CvType.CV_32FC2);
			
			temp_corners.put(0, 0, new double[]{0,0});
			temp_corners.put(1, 0, new double[]{temp.cols(),0});
			temp_corners.put(2, 0, new double[]{temp.cols(),temp.rows()});
			temp_corners.put(3, 0, new double[]{0,temp.rows()});
			
			Core.perspectiveTransform(temp_corners, obj_corners, homography);
			
			Mat img = obj;
			
			Core.line(img, new Point(obj_corners.get(0, 0)), new Point(obj_corners.get(1, 0)), new Scalar(0, 255, 0), 4);
            Core.line(img, new Point(obj_corners.get(1, 0)), new Point(obj_corners.get(2, 0)), new Scalar(0, 255, 0), 4);
            Core.line(img, new Point(obj_corners.get(2, 0)), new Point(obj_corners.get(3, 0)), new Scalar(0, 255, 0), 4);
            Core.line(img, new Point(obj_corners.get(3, 0)), new Point(obj_corners.get(0, 0)), new Scalar(0, 255, 0), 4);
            
           MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatch);
            Mat output = new Mat(obj.rows()*2,obj.cols()*2,Highgui.CV_LOAD_IMAGE_ANYCOLOR);
            Features2d.drawMatches(temp, tempKeyPoints, obj, objKeyPoints, goodMatches, output, newKeyPointColor, newKeyPointColor, new MatOfByte(), 2);
            
            Highgui.imwrite("F:/JAVA/openCV/InterestPoint1/images/box_processed.jpg", img);
            Highgui.imwrite("F:/JAVA/openCV/InterestPoint1/images/match_processed.jpg", output);
            
			
		}
		else{
			System.out.println("no match found");
		}
	}

}
