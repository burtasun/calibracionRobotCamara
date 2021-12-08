#include "pch.h"

using namespace std;
using namespace Eigen;


//glob static pars
constexpr static char _previewWin[]{ "Previews" };
//glob pars
const cv::Scalar RED(0, 0, 255), GREEN(0, 255, 0), WHITE(255,255,255);

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
		struct Input {
			bool inputFile = true;
			vector<string>pathFiles;//imgs
			std::array<int, 2>rowCol_IR_img = { 512,640 };
			bool saveParseImg = true;
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(Input, inputFile, pathFiles, rowCol_IR_img, saveParseImg);
		}input;
		struct CalibratePars {
			vector<double> distParams;//parametros distorsion
			std::array<std::array<double, 3>, 3> intrinsicParams{ 0,0,0,0,0,0,0,0,0 };//matriz proyeccion/rMajor
			std::array<int, 2> widthHeightPattern{ 3,6 };
			cv::Size2i widthHeightPatternSz() { return *(cv::Size*)&this->widthHeightPattern; };
			double dimsSquarePattern = 7.5;
			std::string pathBlobDetectPars;
			bool previewBlobPts = true;
			bool previewPattern = true;
			//TODO mas modos y flags 
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(CalibratePars, distParams, intrinsicParams, widthHeightPattern, dimsSquarePattern, pathBlobDetectPars, previewBlobPts);
		}calibratePars;

		struct HandEyeCalib {
			XyzQuat frame_flange_cam;//output
			vector<string> pathFiles;//imgs
			vector<XyzQuat> frames_rob_flange;
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(HandEyeCalib, frame_flange_cam, pathFiles, frames_rob_flange);
		}handEyeCalib;

		struct ReprojectAndRectify {
			XyzQuat frameReproj;//reproyeccion de referencia / pose 'ideal'
			string pathSaveReprojected;
			bool useExtrincRobotPars = false;//implictamente usaremos frames_rob_flange, caso contrario se asume que dispone de patron
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(ReprojectAndRectify, frameReproj, pathSaveReprojected, useExtrincRobotPars);
		}reprojectAndRectify;

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, modes, input, calibratePars, handEyeCalib, reprojectAndRectify);

		Params() {};
		Params(int argc, char** argv)
		{
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
					return;
				}
				catch (const std::exception& e) {
					cerr << "error al parsear json, cargando valores predeterminados\n\t" << e.what() << '\n';
					break;
				}
			} while (false);

			//errores
			*this = Params();
			fstream fs(Params::rutaJsonPredet, std::ios::out);
			nlohmann::json json; to_json(json, *this);
			fs << json.dump(2);
			cout << "archivo guardado en\n\t" << Params::rutaJsonPredet << '\n';

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
	auto& input = pars.input;
	if (pars.modes.calibrateCam) {
		auto& calibPars = pars.calibratePars;

		cv::Mat1b img;
		int cntValid = 0;
		auto blobDetector = iniBlobDetector(calibPars.pathBlobDetectPars);
		cv::Mat imgShow;
		//captura conjunto de puntos cada imagen
		vector<vector<cv::Point2f> > ptsImgs;
		for (int i = 0; i < input.pathFiles.size(); ++i) {
			//lectura imagen
			if (!readImg(input.pathFiles[i], img, input.rowCol_IR_img[0], input.rowCol_IR_img[1], input.saveParseImg))
				continue;
			//deteccion ptos
			if (calibPars.previewBlobPts) {
				vector<cv::KeyPoint>kp;
				blobDetector->detect(img, kp);
				drawKeypoints(img, kp, imgShow, cv::Scalar{ 0,0,255 });
				previewImg(imgShow, string("previewBlobDetectorKPs, nKps " + to_string(kp.size()) + '\n'));
			}
			//TODO mejorar precision centroide blobs elipticos
			//  extrae base empleando 2 hipotesis 0-180º, y testea contencion en base  convexhull asociado
			vector<cv::Point2f> pts;
			bool found = cv::findCirclesGrid(
				img, calibPars.widthHeightPatternSz(), pts,
				cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING/*mas robusto frente dRadial grandes*/,
				blobDetector);
			if (!found) { cerr << "no se encontraron keypoints en img, " << i << ", continuando"; continue; }
			if (calibPars.previewPattern) {
				cv::cvtColor(img, imgShow, cv::ColorConversionCodes::COLOR_GRAY2BGR);
				cv::drawChessboardCorners(imgShow, calibPars.widthHeightPatternSz(), cv::Mat(pts), true);
				previewImg(imgShow, "patternImg");
			}
			ptsImgs.push_back(pts);
			cntValid++;
		}
	}//calibrateHandEye
	if (pars.modes.calibrateHandEye) {

	}//calibrateHandEye
	if (pars.modes.reprojectAndRectify) {

	}//reprojectAndRectify

	return 0;
}