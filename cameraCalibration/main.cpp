#include "pch.h"

using namespace std;
using namespace Eigen;

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

		struct CalibratePars {
			vector<string>pathFiles;//imgs
			vector<double> distParams;//parametros distorsion
			std::array<std::array<double, 3>, 3> intrinsicParams{ 0,0,0,0,0,0,0,0,0};//matriz proyeccion/rMajor
			std::array<int, 2> widthHeightPattern{ 3,9 };
			double dimsSquarePattern = 7.5;
			//TODO mas modos y flags 
			NLOHMANN_DEFINE_TYPE_INTRUSIVE(CalibratePars, pathFiles, distParams, intrinsicParams, widthHeightPattern, dimsSquarePattern);
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

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, modes, calibratePars, handEyeCalib, reprojectAndRectify);

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
				catch (const std::exception&e) {
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

int main(int argc, char** argv)
{
	//parse input
	auto pars = Parse::Params(argc,argv);
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

	}//calibrateHandEye
	if (pars.modes.calibrateHandEye) {

	}//calibrateHandEye
	if (pars.modes.reprojectAndRectify) {

	}//reprojectAndRectify

	return 0;
}