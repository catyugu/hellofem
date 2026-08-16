import com.comsol.model.GeomInfo;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/**
 * EcTSmBarStationary: 矩形铜母线 稳态 电→热→结构 三场耦合。
 *
 * <p>几何: 3D Block (L x wbb x tbb), 单域。
 * 物理: ec (Terminal V=V0 @ x=L 端面, Ground @ x=0 端面), ht (全外表面对流 htc->T0),
 *       solid (Fixed @ x=0 端面)。
 * 耦合: ElectromagneticHeating (ec->ht 焦耳热), ThermalExpansion (ht->solid, Tref=T0)。
 * 网格: FreeTet, hmax=mh。研究: Stationary。
 *
 * <p>边界识别 (确定性): 用 GeomInfo faceX 采样各面中心, 按坐标平面分类:
 *       x~0 -> ground/fixed, x~L -> terminal, 其余 -> 对流。
 *     面编号是运行时探测的 COMSOL 编号, 与 .mphtxt 边界实体号一致, 驱动据此匹配。
 *
 * <p>元素阶: 显式设为 1 (Prop Order), 与 hellofem P1 解一致。
 * 导出: args[0]=result.txt (V,T,solid.disp), args[1]=mesh.mphtxt,
 *       args[2]=.mph, args[3]=generated_model.java (Save-As-Java)。
 */
public class EcTSmBarStationary {

    public static void main(String[] args) throws Exception {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a
            : new String[]{"result.txt", "mesh.mphtxt", "EcTSmBarStationary.mph", "generated_model.java"};

        Model model = ModelUtil.create("Model");
        model.param().set("L", "0.1[m]", "母线长度");
        model.param().set("wbb", "0.03[m]", "母线宽度");
        model.param().set("tbb", "0.005[m]", "母线厚度");
        model.param().set("V0", "0.02[V]", "端电压");
        model.param().set("T0", "293.15[K]", "环境/无应变参考温度");
        model.param().set("htc", "5[W/(m^2*K)]", "自然对流换热系数");
        model.param().set("mh", "0.002[m]", "最大网格尺寸");
        // Copper
        model.param().set("sig", "5.998e7[S/m]", "铜电导率");
        model.param().set("k", "400[W/(m*K)]", "铜热导率");
        model.param().set("rho", "8700[kg/m^3]", "铜密度");
        model.param().set("Cp", "385[J/(kg*K)]", "铜比热");
        model.param().set("E_Cu", "110[GPa]", "铜杨氏模量");
        model.param().set("nu_Cu", "0.35", "铜泊松比");
        model.param().set("alpha_Cu", "17e-6[1/K]", "铜热膨胀系数");

        String comp = "comp1";
        model.component().create(comp, true);

        // ---- 几何: 矩形块 ----
        com.comsol.model.GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
        g3.create("blk1", "Block");
        g3.feature("blk1").set("size", new String[]{"L", "wbb", "tbb"});
        g3.feature("blk1").set("pos", new String[]{"0", "0", "0"});
        g3.run();
        System.out.println("GEOM_OK");

        // ---- 面识别 (确定性, 按面心坐标分类) ----
        GeomInfo gi = model.component(comp).geom("geom1");
        int nFace = gi.getNFaces();
        java.util.List<Integer> groundFaces = new java.util.ArrayList<Integer>();
        java.util.List<Integer> terminalFaces = new java.util.ArrayList<Integer>();
        java.util.List<Integer> convFaces = new java.util.ArrayList<Integer>();
        for (int f = 1; f <= nFace; f++) {
            double[] pr;
            try { pr = gi.faceParamRange(f); } catch (Exception e) { continue; }
            double[][] pts = null;
            try {
                pts = gi.faceX(f, new double[][]{{(pr[0]+pr[1])/2,
                    pr.length >= 4 ? (pr[2]+pr[3])/2 : 0.5}});
            } catch (Exception e) {
                try { pts = gi.faceX(f, new double[][]{{0.5, 0.5}}); }
                catch (Exception e2) { continue; }
            }
            if (pts == null || pts.length == 0) continue;
            double[] c = pts[0];
            if (Math.abs(c[0]) < 1e-6) groundFaces.add(f);
            else if (Math.abs(c[0] - 0.1) < 1e-6) terminalFaces.add(f);
            else convFaces.add(f);
            System.out.println("FACE " + f + " center=(" + c[0] + "," + c[1] + "," + c[2] + ")");
        }
        int groundFace = groundFaces.get(0);
        int terminalFace = terminalFaces.get(0);
        int[] convArr = new int[convFaces.size()];
        for (int i = 0; i < convFaces.size(); i++) convArr[i] = convFaces.get(i);
        System.out.println("GROUND=" + groundFace + " TERMINAL=" + terminalFace
            + " CONV=" + java.util.Arrays.toString(convArr));
        if (convArr.length != 4) {
            throw new IllegalStateException("expected 4 convection faces, got " + convArr.length);
        }

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
            new String[]{"alpha_Cu", "0", "0", "0", "alpha_Cu", "0", "0", "0", "alpha_Cu"});
        mat.materialModel().create("Enu", "YoungsModulusAndPoissonsRatio");
        mat.propertyGroup("Enu").set("E", new String[][]{{"E_Cu"}});
        mat.propertyGroup("Enu").set("nu", new String[][]{{"nu_Cu"}});
        System.out.println("MAT_OK");

        // ---- 物理场 ----
        model.component(comp).physics().create("ec", "ConductiveMedia", "geom1");
        model.component(comp).physics().create("ht", "HeatTransfer", "geom1");
        model.component(comp).physics().create("solid", "SolidMechanics", "geom1");


        // 电边界
        model.component(comp).physics("ec").create("term1", "Terminal", 2);
        model.component(comp).physics("ec").feature("term1").selection().set(new int[]{terminalFace});
        model.component(comp).physics("ec").feature("term1").set("TerminalType", "Voltage");
        model.component(comp).physics("ec").feature("term1").set("V0", "V0");
        model.component(comp).physics("ec").create("gnd1", "Ground", 2);
        model.component(comp).physics("ec").feature("gnd1").selection().set(new int[]{groundFace});

        // 热边界: 全外表面对流
        model.component(comp).physics("ht").create("hf1", "HeatFluxBoundary", 2);
        model.component(comp).physics("ht").feature("hf1").selection().set(convArr);
        model.component(comp).physics("ht").feature("hf1").set("HeatFluxType", "ConvectiveHeatFlux");
        model.component(comp).physics("ht").feature("hf1").set("minput_temperature_src", "userdef");
        model.component(comp).physics("ht").feature("hf1").set("minput_temperature", "T0");
        model.component(comp).physics("ht").feature("hf1").set("HeatTransferCoefficientType", "UserDef");
        model.component(comp).physics("ht").feature("hf1").set("h", "htc");

        // 多物理场: 焦耳热
        model.component(comp).multiphysics().create("emh1", "ElectromagneticHeating");
        model.component(comp).multiphysics("emh1").set("EMHeat_physics", "ec");
        model.component(comp).multiphysics("emh1").set("Heat_physics", "ht");

        // 结构: Fixed @ ground 端面
        model.component(comp).physics("solid").feature("lemm1").set("E_mat", "from_mat");
        model.component(comp).physics("solid").feature("lemm1").set("nu_mat", "from_mat");
        model.component(comp).physics("solid").create("fix1", "Fixed", 2);
        model.component(comp).physics("solid").feature("fix1").selection().set(new int[]{groundFace});

        // 热膨胀耦合
        model.component(comp).multiphysics().create("te1", "ThermalExpansion");
        model.component(comp).multiphysics("te1").selection().set(new int[]{1});
        model.component(comp).multiphysics("te1").set("Heat_physics", "ht");
        model.component(comp).multiphysics("te1").set("Solid_physics", "solid");
        model.component(comp).multiphysics("te1").set("alpha_mat", "from_mat");
        model.component(comp).multiphysics("te1").set("minput_strainreferencetemperature_src", "userdef");
        model.component(comp).multiphysics("te1").set("minput_strainreferencetemperature", "T0");
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
        model.result().export("data1").set("expr", new String[]{"V", "T", "solid.disp"});
        model.result().export("data1").run();

        model.result().export().create("mesh1", "Mesh");
        model.result().export("mesh1").set("data", "dset1");
        model.result().export("mesh1").set("filename", P[1]);
        model.result().export("mesh1").run();

        model.save(P[2]);
        model.save(P[3], "java");
        System.out.println("EcTSmBar_OK");
    }
}
