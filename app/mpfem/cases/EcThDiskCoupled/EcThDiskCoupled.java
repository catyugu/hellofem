import com.comsol.model.GeomInfo;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/**
 * EcThDiskCoupled: 铜圆盘 稳态 电→热 耦合。
 *
 * <p>几何: 3D Cylinder (R x H), 单域。
 * 物理: ec (Top面V=V0, Bottom面Ground), ht (侧面+顶面 对流 h->T0)。
 * 耦合: ElectromagneticHeating (ec->ht 焦耳热)。
 * 网格: FreeTet, hmax=mh。研究: Stationary。
 *
 * <p>参数: R=0.05, H=0.01, V0=0.01, h=10, T0=293.15
 */
public class EcThDiskCoupled {

    public static void main(String[] args) throws Exception {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a
            : new String[]{"result.txt", "mesh.mphtxt", "EcThDiskCoupled.mph", "generated_model.java"};

        Model model = ModelUtil.create("Model");
        model.param().set("R", "0.05[m]", "圆盘半径");
        model.param().set("H", "0.01[m]", "圆盘厚度");
        model.param().set("V0", "0.01[V]", "顶端电压");
        model.param().set("T0", "293.15[K]", "环境温度");
        model.param().set("htc", "10[W/(m^2*K)]", "对流换热系数");
        model.param().set("mh", "0.005[m]", "最大网格尺寸");
        // Copper
        model.param().set("sig", "5.998e7[S/m]", "铜电导率");
        model.param().set("k", "400[W/(m*K)]", "铜热导率");
        model.param().set("rho", "8700[kg/m^3]", "铜密度");
        model.param().set("Cp", "385[J/(kg*K)]", "铜比热");
        model.param().set("alpha", "17e-6[1/K]", "热膨胀系数");
        model.param().set("E", "110[GPa]", "杨氏模量");
        model.param().set("nu", "0.35", "泊松比");

        String comp = "comp1";
        model.component().create(comp, true);

        // ---- 几何: 圆柱 ----
        com.comsol.model.GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
        g3.create("cyl1", "Cylinder");
        g3.feature("cyl1").set("r", "R");
        g3.feature("cyl1").set("h", "H");
        g3.run();
        System.out.println("GEOM_OK");

        // ---- 面识别 ----
        GeomInfo gi = model.component(comp).geom("geom1");
        int nFace = gi.getNFaces();
        java.util.List<Integer> topFaces = new java.util.ArrayList<Integer>();
        java.util.List<Integer> bottomFaces = new java.util.ArrayList<Integer>();
        java.util.List<Integer> sideFaces = new java.util.ArrayList<Integer>();
        for (int f = 1; f <= nFace; f++) {
            double[] pr;
            try { pr = gi.faceParamRange(f); } catch (Exception e) { continue; }
            double[][] pts = null;
            try {
                pts = gi.faceX(f, new double[][]{{(pr[0]+pr[1])/2,
                    pr.length >= 4 ? (pr[2]+pr[3])/2 : 0.5}});
            } catch (Exception e2) {
                try { pts = gi.faceX(f, new double[][]{{0.5, 0.5}}); }
                catch (Exception e3) { continue; }
            }
            if (pts == null || pts.length == 0) continue;
            double[] c = pts[0];
            double Hval = model.param().evaluate("H");
            if (Math.abs(c[2] - Hval) < 1e-6) topFaces.add(f);
            else if (Math.abs(c[2]) < 1e-6) bottomFaces.add(f);
            else sideFaces.add(f);
        }
        int topFace = topFaces.get(0);
        int bottomFace = bottomFaces.get(0);
        int[] sideArr = new int[sideFaces.size()];
        for (int i = 0; i < sideFaces.size(); i++) sideArr[i] = sideFaces.get(i);
        System.out.println("TOP=" + topFace + " BOTTOM=" + bottomFace + " SIDE=" + java.util.Arrays.toString(sideArr));

        // ---- 材料: Copper ----
        model.component(comp).material().create("mat1", "Common");
        model.component(comp).material("mat1").label("Copper");
        model.component(comp).material("mat1").selection().set(new int[]{1});
        com.comsol.model.Material mat = model.component(comp).material("mat1");
        mat.propertyGroup("def").set("electricconductivity", new String[][]{{"sig"}});
        mat.propertyGroup("def").set("relpermittivity", new String[][]{{"1"}});
        mat.propertyGroup("def").set("thermalconductivity", new String[][]{{"k"}});
        mat.propertyGroup("def").set("density", new String[][]{{"rho"}});
        mat.propertyGroup("def").set("heatcapacity", new String[][]{{"Cp"}});
        mat.propertyGroup("def").set("thermalexpansioncoefficient",
            new String[]{"alpha", "0", "0", "0", "alpha", "0", "0", "0", "alpha"});
        mat.materialModel().create("Enu", "YoungsModulusAndPoissonsRatio");
        mat.propertyGroup("Enu").set("E", new String[][]{{"E"}});
        mat.propertyGroup("Enu").set("nu", new String[][]{{"nu"}});
        System.out.println("MAT_OK");

        // ---- 物理场 ----
        model.component(comp).physics().create("ec", "ConductiveMedia", "geom1");
        model.component(comp).physics().create("ht", "HeatTransfer", "geom1");

        // 电边界
        model.component(comp).physics("ec").create("term1", "Terminal", 2);
        model.component(comp).physics("ec").feature("term1").selection().set(new int[]{topFace});
        model.component(comp).physics("ec").feature("term1").set("TerminalType", "Voltage");
        model.component(comp).physics("ec").feature("term1").set("V0", "V0");
        model.component(comp).physics("ec").create("gnd1", "Ground", 2);
        model.component(comp).physics("ec").feature("gnd1").selection().set(new int[]{bottomFace});

        // 热边界: 侧面对流
        model.component(comp).physics("ht").create("hf1", "HeatFluxBoundary", 2);
        model.component(comp).physics("ht").feature("hf1").selection().set(sideArr);
        model.component(comp).physics("ht").feature("hf1").set("HeatFluxType", "ConvectiveHeatFlux");
        model.component(comp).physics("ht").feature("hf1").set("minput_temperature_src", "userdef");
        model.component(comp).physics("ht").feature("hf1").set("minput_temperature", "T0");
        model.component(comp).physics("ht").feature("hf1").set("HeatTransferCoefficientType", "UserDef");
        model.component(comp).physics("ht").feature("hf1").set("h", "htc");

        // 多物理场: 焦耳热
        model.component(comp).multiphysics().create("emh1", "ElectromagneticHeating");
        model.component(comp).multiphysics("emh1").set("EMHeat_physics", "ec");
        model.component(comp).multiphysics("emh1").set("Heat_physics", "ht");

        System.out.println("PHYS_OK");

        // ---- 网格 ----
        model.component(comp).mesh().create("mesh1");
        com.comsol.model.MeshFeature ftet1 = model.component(comp).mesh("mesh1").create("ftet1", "FreeTet");
        com.comsol.model.MeshFeature size1 = ftet1.create("size1", "Size");
        size1.set("custom", "on");
        size1.set("hmax", "mh");
        size1.set("hmin", "mh/2");
        size1.set("hcurve", 0.2);
        size1.set("hgrad", 1.5);
        model.component(comp).mesh("mesh1").run();
        System.out.println("MESH_OK");

        // ---- 研究 ----
        model.study().create("std1");
        model.study("std1").create("stat", "Stationary");
        model.study("std1").createAutoSequences("stat");
        model.study("std1").run();
        System.out.println("STUDY_OK");

        // ---- 导出 ----
        model.result().export().create("data1", "Data");
        model.result().export("data1").set("data", "dset1");
        model.result().export("data1").set("filename", P[0]);
        model.result().export("data1").set("expr", new String[]{"V", "T"});
        model.result().export("data1").run();

        model.result().export().create("mesh1", "Mesh");
        model.result().export("mesh1").set("data", "dset1");
        model.result().export("mesh1").set("filename", P[1]);
        model.result().export("mesh1").run();

        model.save(P[2]);
        model.save(P[3], "java");
        System.out.println("EcThDisk_OK");
    }
}
