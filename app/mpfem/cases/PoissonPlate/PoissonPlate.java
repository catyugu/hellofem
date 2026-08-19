import com.comsol.model.GeomInfo;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/**
 * PoissonPlate: 纯电场 静电学 Poisson问题（3D薄板）。
 *
 * <p>几何: 3D Block (L x W x T), 单域。
 * 物理: ec (Front V=V0, Back Ground, 其余绝缘)。
 * 网格: FreeTet, hmax=mh。研究: Stationary。
 *
 * <p>用于验证单物理场直线问题，无耦合。
 */
public class PoissonPlate {

    public static void main(String[] args) throws Exception {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a
            : new String[]{"result.txt", "mesh.mphtxt", "PoissonPlate.mph", "generated_model.java"};

        Model model = ModelUtil.create("Model");
        model.param().set("L", "0.1[m]", "板长度");
        model.param().set("W", "0.05[m]", "板宽度");
        model.param().set("T", "0.001[m]", "板厚度");
        model.param().set("V0", "1.0[V]", "前端电压");
        model.param().set("sig", "1[S/m]", "电导率");
        model.param().set("mh", "0.01[m]", "最大网格尺寸");

        String comp = "comp1";
        model.component().create(comp, true);

        // ---- 几何: 3D Block ----
        com.comsol.model.GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
        g3.create("blk1", "Block");
        g3.feature("blk1").set("size", new String[]{"L", "W", "T"});
        g3.feature("blk1").set("pos", new String[]{"0", "0", "0"});
        g3.run();
        System.out.println("GEOM_OK");

        // ---- 面识别 ----
        GeomInfo gi = model.component(comp).geom("geom1");
        int nFace = gi.getNFaces();
        java.util.List<Integer> frontFaces = new java.util.ArrayList<Integer>();
        java.util.List<Integer> backFaces = new java.util.ArrayList<Integer>();
        java.util.List<Integer> sideFaces = new java.util.ArrayList<Integer>();
        double Lval = model.param().evaluate("L");
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
            if (Math.abs(c[0]) < 1e-6) frontFaces.add(f);
            else if (Math.abs(c[0] - Lval) < 1e-6) backFaces.add(f);
            else sideFaces.add(f);
        }
        int frontFace = frontFaces.get(0);
        int backFace = backFaces.get(0);
        int[] sideArr = new int[sideFaces.size()];
        for (int i = 0; i < sideFaces.size(); i++) sideArr[i] = sideFaces.get(i);
        System.out.println("FRONT=" + frontFace + " BACK=" + backFace + " SIDE=" + java.util.Arrays.toString(sideArr));

        // ---- 材料 ----
        model.component(comp).material().create("mat1", "Common");
        model.component(comp).material("mat1").selection().set(new int[]{1});
        com.comsol.model.Material mat = model.component(comp).material("mat1");
        mat.propertyGroup("def").set("electricconductivity", new String[][]{{"sig"}});
        System.out.println("MAT_OK");

        // ---- 物理场 ----
        model.component(comp).physics().create("ec", "ConductiveMedia", "geom1");

        // 电边界
        model.component(comp).physics("ec").create("term1", "Terminal", 2);
        model.component(comp).physics("ec").feature("term1").selection().set(new int[]{frontFace});
        model.component(comp).physics("ec").feature("term1").set("TerminalType", "Voltage");
        model.component(comp).physics("ec").feature("term1").set("V0", "V0");
        model.component(comp).physics("ec").create("gnd1", "Ground", 2);
        model.component(comp).physics("ec").feature("gnd1").selection().set(new int[]{backFace});

        System.out.println("PHYS_OK");

        // ---- 网格 ----
        model.component(comp).mesh().create("mesh1");
        com.comsol.model.MeshFeature ftet1 = model.component(comp).mesh("mesh1").create("ftet1", "FreeTet");
        com.comsol.model.MeshFeature size1 = ftet1.create("size1", "Size");
        size1.set("custom", "on");
        size1.set("hmax", "mh");
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
        model.result().export("data1").set("expr", new String[]{"V"});
        model.result().export("data1").run();

        model.result().export().create("mesh1", "Mesh");
        model.result().export("mesh1").set("data", "dset1");
        model.result().export("mesh1").set("filename", P[1]);
        model.result().export("mesh1").run();

        model.save(P[2]);
        model.save(P[3], "java");
        System.out.println("PoissonPlate_OK");
    }
}
