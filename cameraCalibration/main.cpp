#include "pch.h"

using namespace std;
using namespace Eigen;


//glob static pars
constexpr static char _previewWin[]{ "Previews" };
//glob pars
const cv::Scalar RED(0, 0, 255), GREEN(0, 255, 0), WHITE(255, 255, 255);

struct XyzQuat {
	float_t x = 0, y = 0, z = 0, qx = 0, qy = 0, qz = 0, qw = 1;
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(XyzQuat, x, y, z, qx, qy, qz, qw);
	M4 toM4() const {
		M4 ret(M4::Identity());
		ret.topRightCorner<3, 1>() = V3(x, y, z);
		ret.topLeftCorner<3, 3>() = Eigen::Quaternion<float_t>(qw, qx, qy, qz).toRotationMatrix();
		return ret;
	}
	XyzQuat fromM4(M4& T) const {
		Eigen::Quaternion<float_t> q(T.topLeftCorner<3, 3>());
		XyzQuat ret{ T(0,3),T(1,3),T(2,3),q.x(), q.y(), q.z(), q.w() };
		return ret;
	}
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(XyzQuat, x, y, z, qx, qy, qz, qw);
};//hand-eye
//parse pars
namespace Parse
{
	struct Params {
		constexpr static const char rutaJsonPredet[] = ".\\Parametros.json";
	public:
		//varios modos
		struct Modes {
			//	calibracion camara
			bool calibrateCam = true;//si no se activa se carga de saveCalib
			//	calibracion hand-eye
			bool calibrateHandEye = false;//si no se activa se carga de saveCalib
			//	reproyeccion y rectificacion
			bool reprojectAndRectify = false;//opcionalmente empleando extrinseco robot
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(Modes,
				calibrateCam, reprojectAndRectify, calibrateHandEye);
		}modes;
		struct Input {//TODO mas tipos de inputs & enlazar con SDK Flir
			bool inputFile = true;
			vector<string>pathFiles;//imgs
			std::array<int, 2>rowCol_IR_img = { 512,640 };
			bool saveParseImg = true;
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(Input, inputFile, pathFiles, rowCol_IR_img, saveParseImg);
		}input;
		struct CalibratePars {
			vector<double> distParams;//parametros distorsion
			std::array<std::array<double, 3>, 3> intrinsicParams{ 0,0,0,0,0,0,0,0,0 };//matriz proyeccion/rMajor
			decltype(intrinsicParams) intrinsicParamsGuess = { 0,0,0,0,0,0,0,0,0 };
			bool guessUse = false;
			cv::Mat getIntrinsicParsCV() const { cv::Mat ret(3, 3, CV_64F); memcpy(ret.data, intrinsicParams.data(), sizeof(double) * 9); return ret; };
			std::array<int, 2> widthHeightPattern{ 3,6 };
			cv::Size2i widthHeightPatternSz() const { return *(cv::Size*)&this->widthHeightPattern; };
			double dimsSquarePattern = 7.5;
			std::string pathBlobDetectPars;
			struct Prevs {
				bool previewBlobPts = true;
				bool previewPattern = true;
				bool previewUndistort = true;
				NLOHMANN_DEFINE_TYPE_INTRUSIVE(Prevs, previewBlobPts, previewPattern, previewUndistort);
			}prevs;
			int minImgsCalib = 20;
			//TODO mas modos y flags 
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(CalibratePars, distParams, intrinsicParams, intrinsicParamsGuess, guessUse, widthHeightPattern, dimsSquarePattern, pathBlobDetectPars, prevs, minImgsCalib);
		}calibratePars;
		struct HandEyeCalib {
			XyzQuat frame_flange_cam;//output
			string pathFramesRob;
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(HandEyeCalib, frame_flange_cam, pathFramesRob);
		}handEyeCalib;
		struct ReprojectAndRectify {
			XyzQuat frameReproj;//reproyeccion de referencia / pose 'ideal'
			string pathSaveReprojected = "";
			bool useExtrincRobotPars = false;//implictamente usaremos frames_rob_flange, caso contrario se asume que dispone de patron
			string pathImageReference = "";
			bool previewReprojected = true;
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(ReprojectAndRectify, frameReproj, pathSaveReprojected, useExtrincRobotPars, pathImageReference, previewReprojected);
		}reprojectAndRectify;

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, modes, input, calibratePars, handEyeCalib, reprojectAndRectify);

		std::string loadParamsPath;

		Params() {};
		Params(int argc, char** argv)
		{
			loadParamsPath = rutaJsonPredet;
			do {
				if (argc < 2) {
					cerr << "no se ha especificado ningun parametro, empleando parametros predefinidos\n";
					break;
				}
				fstream fs(argv[1], std::ios::in);
				if (!fs.is_open()) {
					cout << "no se pudo abrir: " << argv[1] << "\nempleando paramretros por defecto\n";
					break;
				}
				nlohmann::json json;
				try
				{
					fs >> json;
					from_json(json, *this);
					cout << "parametros parseados de: " << argv[1] << '\n';
					cout << json.dump(2) << '\n';
					loadParamsPath = argv[1];
					return;
				}
				catch (const std::exception& e) {
					cerr << "error al parsear json, cargando valores predeterminados\n\t" << e.what() << '\n';
					break;
				}
			} while (false);

			//errores
			*this = Params();
			saveParams(Params::rutaJsonPredet);
		}
		bool saveParams(const std::string& path = rutaJsonPredet) {
			fstream fs(path, ios::out);
			if (!fs.is_open()) { cerr << "no se pudo guardar parametros\n"; return false; }
			nlohmann::json json; to_json(json, *this);
			fs << json.dump(2);
			cout << "archivo guardado en\n\t" << path << '\n';
		}
	}params;
};//ns parse


//devuelve imagen de 16 bits raw y 8 bits escalada minMax
bool readRawImg_DispImg(const std::string& path, const int nRows, const int nCols, cv::Mat& img)
{
	const int offRead = 28;//cabecera hardCodeada, en imagenes sueltas 16
	std::fstream fs(path, std::ios::in | std::ios::binary | std::ios::end);
	if (!fs.is_open()) {
		cerr << "no se pudo abrir " << path << '\n'; return false;
	}
	fs.seekg(0, std::ios::end);
	auto endPos = fs.tellg();
	fs.seekg(offRead, std::ios::beg);
	//cout << "endPos " << endPos << "  fs.tellg() " << fs.tellg() << '\n';
	img = cv::Mat(nRows, nCols, CV_16UC1);
	if (!(img.rows * img.cols * sizeof(short) == endPos - fs.tellg())) {
		cerr << "tamanio de buffer imagen incoherente, " << nRows << 'x' << nCols << 'x' << to_string(sizeof(short)) << "!=" << to_string(int(endPos - fs.tellg())) << '\n';
		return false;
	}
	fs.read((char*)img.data, 2 * img.rows * img.cols);
	return true;
};

void minMax_to_byte(const cv::Mat& in, cv::Mat1b& out) {
	//minMax
	double minV = 0, maxV = 0; cv::minMaxIdx(in, &minV, &maxV);
	//scale to minMax
	double alpha = (0 - 255.) / (minV - maxV);
	double beta = 0 - minV * alpha;
	cv::Mat1d imgD;
	in.convertTo(imgD, CV_64F, alpha, beta);//v * alpha + beta
	imgD.convertTo(out, CV_8U);
}


bool readImg(const std::string& path, cv::Mat1b& img, const int nRows = 0, const int nCols = 0, const bool saveParsedImg = false) {
	auto pos = path.find_last_of('.');
	if (pos == std::string::npos) {
		cerr << "imagen sin extension\n\t" << path << "\n";
		return false;
	}
	auto ext = string(path.begin() + pos + 1, path.end());
	if (!strcmp(ext.data(), "bin")) {
		cv::Mat imgRaw;
		if (!nRows || !nCols) { cerr << "rows Cols sin especificar\n"; return false; }
		if (!readRawImg_DispImg(path, nRows, nCols, imgRaw)) { cerr << "error al leer archivo\n\t" << path << '\n'; return false; }
		minMax_to_byte(imgRaw, img);
	}
	else
		img = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (saveParsedImg && img.data) {
		std::string pathW = string(path.begin(), path.begin() + pos) + ".tiff";
		cv::imwrite(pathW, img);
	}
	return img.data != 0;
}
decltype(cv::SimpleBlobDetector::create()) iniBlobDetector(const std::string path) {
	auto blobDetect = cv::SimpleBlobDetector::create();
	blobDetect->read(path);
	return blobDetect;
}

void previewImg(const cv::Mat& img, std::string msg = "", bool wait = true) {
	if (!msg.empty()) {
		std::cout << msg;
		int baseLine = 0;
		cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseLine);
		cv::Point textOrigin(img.cols - 2 * textSize.width - 10, img.rows - 2 * baseLine - 10);
		cv::putText(img, msg, textOrigin, 1, 1, GREEN);
	}
	cv::imshow(_previewWin, img);
	if (wait)
		cv::waitKey();
}



using namespace cv;




template<typename T>
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3_<T>>& corners)
{
	corners.clear();
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.emplace_back((2 * j + i % 2) * squareSize, i * squareSize, 0);
}

//suma cuadratica distancias respecto a final
double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	size_t totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (size_t i = 0; i < objectPoints.size(); ++i)
	{
		projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		err = norm(imagePoints[i], imagePoints2, NORM_L2);

		size_t n = objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err * err / n);
		totalErr += err * err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

//TODO integracion parcial, 
bool runCalibration(
	const cv::Size& patternSize, const double dimsSquarePattern,
	cv::Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs, double& totalAvgErr, vector<Point3f>& newObjPoints, bool guessUse)
{
	double grid_width = dimsSquarePattern * (patternSize.width - 1);
	//! [fixed_aspect]
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	//if (s.flag & CALIB_FIX_ASPECT_RATIO)
	//		cameraMatrix.at<double>(0, 0) = s.aspectRatio;
	//! [fixed_aspect]
	//if (s.useFisheye) {
	//	distCoeffs = Mat::zeros(4, 1, CV_64F);
	//}
	//else {
	distCoeffs = Mat::zeros(8, 1, CV_64F);
	//}

	vector<vector<Point3f> > objectPoints(1);
	calcBoardCornerPositions(patternSize, dimsSquarePattern, objectPoints[0]);
	objectPoints[0][patternSize.width - 1].x = objectPoints[0][0].x + grid_width;
	newObjPoints = objectPoints[0];

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	//Find intrinsic and extrinsic camera parameters
	double rms;


	int iFixedPoint = -1;

	int flag = 0;//TODO integrar
	{
		if(guessUse)
			flag |= CALIB_USE_INTRINSIC_GUESS;//////////////
		flag |= CALIB_FIX_PRINCIPAL_POINT;
		flag |= CALIB_ZERO_TANGENT_DIST;
		flag |= CALIB_FIX_ASPECT_RATIO;
		//flag |= CALIB_FIX_K1;
		flag |= CALIB_FIX_K2;
		flag |= CALIB_FIX_K3;
		flag |= CALIB_FIX_K4;
		flag |= CALIB_FIX_K5;
		flag |= CALIB_FIX_K6;
	}
	cout << "regresion parametros camara\n";
	rms = cv::calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
		cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
		flag | CALIB_USE_LU,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS , 30, DBL_EPSILON));

	//consistencia numerica !nan o inf
	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	objectPoints.clear();
	objectPoints.resize(imagePoints.size(), newObjPoints);
	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
		distCoeffs, reprojErrs);

	cout << "Error calibracion err, suma cuadratica distancias a reproyectados " << totalAvgErr << '\n';

	return ok;
}

void storeCalib(Parse::Params::CalibratePars& pars,
	Mat& cameraMatrix, Mat& distCoeffs)
{
	constexpr auto eigenMap_CV_d = [](cv::Mat& mat) {//TODO a cabecera CV
		return Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
			(double*)mat.data, mat.rows, mat.cols);
	};
	memcpy(pars.intrinsicParams.data(), cameraMatrix.data, 9 * sizeof(double));
	pars.distParams.resize(distCoeffs.rows * distCoeffs.cols);
	memcpy(pars.distParams.data(), distCoeffs.data, pars.distParams.size() * sizeof(double));
}

//extraccion ptos de referencia(2d) segun parametros, desde imagenes
template<typename T>
int getImagesPts(
	const Parse::Params& pars,
	vector<vector<cv::Point_<T>>>& ptsImgs,
	cv::Size& imgSize,
	vector<int>& idValids)
{
	auto& input = pars.input;
	auto& calibPars = pars.calibratePars;
	ptsImgs.resize(input.pathFiles.size());
	int cntValid = 0;
	int nThreads = 1;

	bool mpReadImg = !(calibPars.prevs.previewPattern || calibPars.prevs.previewBlobPts);
	if (mpReadImg)
		nThreads = omp_get_max_threads();
	cv::Mat imgShow;

	omp_set_num_threads(nThreads);
#pragma omp parallel
	{
		cv::Mat1b img;
		auto blobDetector = iniBlobDetector(calibPars.pathBlobDetectPars);
#pragma omp for
		for (int i = 0; i < input.pathFiles.size(); ++i) {
			//lectura imagen
			if (!readImg(input.pathFiles[i], img, input.rowCol_IR_img[0], input.rowCol_IR_img[1], input.saveParseImg))
				continue;
			imgSize.width = img.cols; imgSize.height = img.rows;
			//deteccion ptos
			if (calibPars.prevs.previewBlobPts) {
				vector<cv::KeyPoint>kp;
				blobDetector->detect(img, kp);
				drawKeypoints(img, kp, imgShow, cv::Scalar{ 0,0,255 });
				previewImg(imgShow, string("previewBlobDetectorKPs, nKps " + to_string(kp.size()) + '\n'));
			}
			//TODO mejorar precision centroide blobs elipticos
			//  extrae base empleando 2 hipotesis 0-180º, y testea contencion en base  convexhull asociado
			vector<cv::Point_<T>> pts;
			bool found = cv::findCirclesGrid(
				img, calibPars.widthHeightPatternSz(), pts,
				cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING/*mas robusto frente dRadial grandes*/,
				blobDetector);
			if (!found) { cerr << "no se encontraron keypoints en img, " << i << ", continuando\n"; continue; }
			if (calibPars.prevs.previewPattern) {
				cv::cvtColor(img, imgShow, cv::ColorConversionCodes::COLOR_GRAY2BGR);
				cv::drawChessboardCorners(imgShow, calibPars.widthHeightPatternSz(), cv::Mat(pts), true);
				previewImg(imgShow, "patternImg");
			}
			ptsImgs[i] = pts;
			cntValid++;
		}//puntos imagenes validas
	}
	auto itAct = ptsImgs.begin();
	auto it = itAct;
	idValids.reserve(cntValid);
	for (; it != ptsImgs.end(); ++it) {
		if (it->size()) {
			it->swap(*itAct);
			itAct++;
			idValids.push_back(std::distance(ptsImgs.begin(), it));
		}
	}
	assert(std::distance(ptsImgs.begin(), itAct) == cntValid);
	ptsImgs.erase(itAct, ptsImgs.end());
	return cntValid;
}

//X\tY\tZ\tA\tB\tC | R=R_z,a R_z,b R_z,c
//961.684875	-1342.46826	370.749390	-9.80989170	87.9560623	49.5647163
vector<M4> parseKukaPoses(const std::string& path) {
	constexpr auto convKukaToM4 = [](array<double, 6>& nums) {
		M4 out(M4::Identity());
		out.topLeftCorner<3, 3>() = M3(
			Eigen::AngleAxis<M4::Scalar>(deg2rad(nums[3]), V3::UnitZ()) *
			Eigen::AngleAxis<M4::Scalar>(deg2rad(nums[4]), V3::UnitY()) *
			Eigen::AngleAxis<M4::Scalar>(deg2rad(nums[5]), V3::UnitX()));
		for (int i = 0; i < 3; ++i) out(i, 3) = nums[i];
		return out;
	};
	fstream fs(path, ios::in);
	if (!fs.is_open()) { cerr << "parseKukaPoses, no se pudo abrir\n\t" << path << '\n'; return {}; }
	vector<M4>out;
	char buffer[10000];
	std::string_view strv(buffer, 10000);
	std::array<double, 6>nums;
	size_t off1 = 0, off2 = 0;
	while (!fs.eof()) {
		fs.getline(buffer, 10000);
		off1 = off2 = 0;
		for (int i = 0; i < 6; ++i) {
			off1 = strv.find('\t', off2);
			if (off1 == string::npos && i != 5)
				break;//no es
			nums[i] = atof(buffer + off2);
			off2 = off1 + 1;
			if (i == 5)
				out.push_back(convKukaToM4(nums));
		}
	}
	return out;
}


void M4_to_rvec_tvec(const M4& m, cv::Mat& rvec, cv::Mat& tvec) {
	tvec = cv::Mat(1, 3, CV_64F);
	rvec = cv::Mat(3, 3, CV_64F);
	for (int i = 0; i < 3; ++i)
		tvec.at<double>(i) = m(i, 3);
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>((double*)rvec.data) = m.topLeftCorner<3, 3>().cast<double>();
	/*Eigen::Map<Vector3d>((double*)rvec.data) =
		Eigen::AngleAxisd(m.topLeftCorner<3, 3>().cast<double>()).axis()*
		Eigen::AngleAxisd(m.topLeftCorner<3, 3>().cast<double>()).angle();*/
}

void invertRT_CV(const cv::Mat& r, const cv::Mat& t, cv::Mat& rinv, cv::Mat& tinv) {
	//rinv=r^T
	//tinv=-r^T t
	if (r.rows == 3 && r.cols == 3) {
		cv::transpose(r, rinv);
		tinv = -rinv * t;
	}
	else {
		cv::Mat rotTmp;
		rinv = -r;
		cv::Rodrigues(rinv, rotTmp);
		tinv = -rotTmp * t;
	}
}

int main(int argc, char** argv)
{
	//parse input
	auto pars = Parse::Params(argc, argv);
	//calibracion camara
	// 	   lectura imagen //TODO asociar a frameGrabberExterno?
	// 			deteccion ptos
	// 			decodificar patron / asociar ptos a frame patron
	// 	   calibrar
	//			regresion distorsion y matriz proyeccion
	//			guardar parametros
	//calibracion robot-camara / hand-eye //TODO integrar en aplicacion comunicando con robot?
	// 
	//reproyeccion extrinseca robot o extrinseca patron
	if (pars.modes.calibrateCam) {
		do {
			vector < vector<cv::Point2f>>ptsImgs;
			cv::Size imgSize;
			vector<int>idValids;
			//captura conjunto de puntos cada imagen
			int cntValid = getImagesPts(pars, ptsImgs, imgSize, idValids);
			if (!cntValid)
				break;
			auto& calibPars = pars.calibratePars;

			if (calibPars.minImgsCalib > cntValid) {
				cerr << "no se puede calibrar, numero minimo de imagenes validas es " << calibPars.minImgsCalib << ", se han obtenido " << ptsImgs.size() << " validas.\n";
				break;
			}
			cv::Mat cameraMatrix = calibPars.getIntrinsicParsCV();
			cv::Mat distCoeffs;
			vector<cv::Mat>rvecs, tvecs;
			vector<float>reprojErrs; double totErr;
			vector<cv::Point3f>ptsReproj;
			bool calibOk = runCalibration(
				calibPars.widthHeightPatternSz(), calibPars.dimsSquarePattern, imgSize,
				cameraMatrix, distCoeffs, ptsImgs, rvecs, tvecs, reprojErrs, totErr, ptsReproj, calibPars.guessUse);
			if (!calibOk) { cerr << "no se ha podido calibrar la camara\n"; break; }
			storeCalib(calibPars, cameraMatrix, distCoeffs);
			pars.saveParams(pars.loadParamsPath);
			if (calibPars.prevs.previewUndistort) {
				cv::Mat1b img, imShow, imboth;
				for (auto i : idValids) {
					readImg(pars.input.pathFiles[i], img, pars.input.rowCol_IR_img[0], pars.input.rowCol_IR_img[1]);
					cv::undistort(img, imShow, cameraMatrix, distCoeffs);
					cv::hconcat(img, imShow, imboth);
					previewImg(imboth, std::string("orig     undistort" + to_string(i)));
				}
			}
		} while (false);
	}//calibrateCam

	if (pars.modes.calibrateHandEye) {
		do {
			//parse robot frames poses
			auto& he = pars.handEyeCalib;
			vector<cv::Mat> rvecsRob, tvecsRob;//T_rob_brida
			{
				auto framesHe = parseKukaPoses(he.pathFramesRob);
				if (framesHe.empty()) {
					cerr << "el numero de poses robot no coincide con las imagenes, " << framesHe.size() << '\\' << pars.input.pathFiles.size() << '\n';
					break;
				}
				rvecsRob.resize(framesHe.size()); tvecsRob.resize(framesHe.size());
				for (int i = 0; i < framesHe.size(); ++i)
					M4_to_rvec_tvec(framesHe[i], rvecsRob[i], tvecsRob[i]);
			}

			//for (auto& f : framesHe)cout << "\n\n" << f; cout << '\n';
			//pts imgs
			int nImgsValid = 0;
			cv::Mat1b img;
			cv::Size imgSize;
			vector<int>idValids;
			vector<vector<cv::Point2d>>ptsImgs;
			int cntValid = getImagesPts(pars, ptsImgs, imgSize, idValids);
			if (!cntValid)
				break;
			//obtencion poses T_cam_ref
			auto camMat = pars.calibratePars.getIntrinsicParsCV();
			vector<cv::Mat>rvecs_cam_ref, tvecs_cam_ref;//T_cam_ref
			vector<cv::Point3d> ptsRef;//ptos frame pattern
			calcBoardCornerPositions(pars.calibratePars.widthHeightPatternSz(), pars.calibratePars.dimsSquarePattern, ptsRef);
			for (int i = 0; i < cntValid; ++i) {
				cv::Mat rvecCam, tvecCam;
				if (!cv::solvePnP(ptsRef, ptsImgs[i], camMat, pars.calibratePars.distParams, rvecCam, tvecCam, false, cv::SOLVEPNP_ITERATIVE)) {
					cerr << "Improbable!\n";
					continue;
				}
				cv::Mat r_cam; cv::Rodrigues(rvecCam, r_cam);
				rvecs_cam_ref.push_back(r_cam);
				tvecs_cam_ref.push_back(tvecCam);
				//{
				//	cv::Mat rinv, tinv;
				//	invertRT_CV(r_cam, tvecCam, rinv, tinv);
				//	cout << tinv.at<double>(0) << ' ' << tinv.at<double>(1) << ' ' << tinv.at<double>(2) << '\n';
				//	cout << tvecCam.at<double>(0) << ' ' << tvecCam.at<double>(1) << ' ' << tvecCam.at<double>(2) << "\n\n";
				//}
			}
			//hand-eye
			cv::Mat rvecsFlange_Cam, tvecsFlange_Cam;//T_flange_cam
			//consistencia numero roto-traslaciones ambos sets
			//	descarte roto-traslaciones robot sin correspondientes
			{
				if (idValids.back() > rvecsRob.size()) {
					cerr << "mas poses capturadas validas, que poses robot proporcionadas\n"; break;
				}
				int idAct = 0;
				for (int i = 0; i < idValids.size(); ++i) {
					std::swap(rvecsRob[idValids[i]], rvecsRob[i]);
					std::swap(tvecsRob[idValids[i]], tvecsRob[i]);
				}
				rvecsRob.erase(rvecsRob.begin() + idValids.size(), rvecsRob.end());
				tvecsRob.erase(tvecsRob.begin() + idValids.size(), tvecsRob.end());
			}
			vector<V3>ts;
			array<int, 5>ids{ 0,1,2,4 };
			for (int i = 0; i < ids.size(); ++i) {
				cv::calibrateHandEye(rvecsRob, tvecsRob, rvecs_cam_ref, tvecs_cam_ref, rvecsFlange_Cam, tvecsFlange_Cam, (cv::HandEyeCalibrationMethod)ids[i]);// cv::HandEyeCalibrationMethod::CALIB_HAND_EYE_DANIILIDIS);
				//cout << "rvecsFlange_Cam =\n" << rvecsFlange_Cam << "\n\n";
				cout << "tvecsFlange_Cam =\n" << tvecsFlange_Cam << "\n\n";
				//{
				//	cv::Mat rinv, tinv; invertRT_CV(rvecsFlange_Cam, tvecsFlange_Cam, rinv, tinv);
				//	cout << "rinv=\n" << rinv << "\n\n";
				//	cout << "tinv =\n" << tinv << "\n\n";
				//}
				ts.push_back(Eigen::Map<Eigen::Vector3d>((double*)tvecsFlange_Cam.data).cast<float>());
			}
			//evaluar distancia entre metodos
			Mx dists(Mx::Zero(ts.size(), ts.size()));
			for (int i = 0; i < ts.size(); ++i)
				for (int j = i + 1; j < ts.size(); ++j)
					dists(i, j) = dists(j, i) = (ts[i] - ts[j]).norm();
			V3 aver(V3::Zero()); for (auto& t : ts)aver += t; aver /= float(ts.size());
			cout << "ts\n" << dists << "\naver "<<aver.transpose()<<'\n';
		} while (false);
	}//calibrateHandEye

	if (pars.modes.reprojectAndRectify) {

	}//reprojectAndRectify

	return 0;
}