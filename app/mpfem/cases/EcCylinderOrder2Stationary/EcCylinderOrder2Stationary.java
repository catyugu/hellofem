import com.comsol.model.GeomInfo;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/**
 * EcCylinderOrder2Stationary: 3D 空心圆柱（同轴圆筒）导电稳态，二阶几何 + 二阶单元。
 *
 * <p>几何: 外圆柱 r2 减去内圆柱 r1, 高 hcyl。内、外表面都是曲面, 因此 COMSOL
 * 的 Mesh 导出写的是二阶几何单元 (edg2/tri2/tet2), 即等参 P2 几何。
 *
 * <p>物理: ec (内表面 Terminal V=V0, 外表面 Ground, 端面默认绝缘)。
 * 网格: FreeTet, hmax=mh。研究: Stationary。
 *
 * <p>解析参照 (验证要点): 端面绝缘时电位只沿径向变化,
 *       V(r) = V0 * ln(r/r2) / ln(r1/r2)
 * 于是 E_r = -V0 / (r * ln(r1/r2)), 且 V 的等值面是同轴圆柱面。
 * 网格是二阶、单元也是二阶, 所以这个解析解可以同时检验二阶几何映射
 * (曲面上的边中点必须落在真实圆柱面上) 与二阶单元。
 *
 * <p>导出: args[0]=result.txt (V), args[1]=mesh.mphtxt, args[2]=.mph,
 *       args[3]=generated_model.java (Save-As-Java)。
 */
public class EcCylinderOrder2Stationary {

    public static void main(String[] args) throws Exception {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a
            : new String[]{"result.txt", "mesh.mphtxt", "EcCylinderOrder2Stationary.mph",
                "generated_model.java"};

        Model model = ModelUtil.create("Model");
        model.param().set("r1", "0.01[m]", "内半径");
        model.param().set("r2", "0.03[m]", "外半径");
        model.param().set("hcyl", "0.02[m]", "高度");
        model.param().set("V0", "1[V]", "内表面电压");
        model.param().set("sig", "1[S/m]", "电导率");
        model.param().set("mh", "0.003[m]", "最大网格尺寸");

        String comp = "comp1";
        model.component().create(comp, true);

        // ---- 几何: 外圆柱 - 内圆柱 ----
        com.comsol.model.GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
        g3.create("cyl1", "Cylinder");
        g3.feature("cyl1").set("r", "r2");
        g3.feature("cyl1").set("h", "hcyl");
        g3.feature("cyl1").set("pos", new String[]{"0", "0", "0"});
        g3.create("cyl2", "Cylinder");
        g3.feature("cyl2").set("r", "r1");
        g3.feature("cyl2").set("h", "hcyl");
        g3.feature("cyl2").set("pos", new String[]{"0", "0", "0"});
        g3.create("dif1", "Difference");
        g3.feature("dif1").selection("input").set(new String[]{"cyl1"});
        g3.feature("dif1").selection("input2").set(new String[]{"cyl2"});
        g3.run();
        System.out.println("GEOM_OK");

        // ---- 面识别: 按采样点到轴的距离分类 ----
        GeomInfo gi = model.component(comp).geom("geom1");
        int nFace = gi.getNFaces();
        double r1v = model.param().evaluate("r1");
        double r2v = model.param().evaluate("r2");
        java.util.List<Integer> innerFaces = new java.util.ArrayList<Integer>();
        java.util.List<Integer> outerFaces = new java.util.ArrayList<Integer>();
        java.util.List<Integer> otherFaces = new java.util.ArrayList<Integer>();
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
            double rr = Math.sqrt(c[0]*c[0] + c[1]*c[1]);
            if (Math.abs(rr - r1v) < 1e-6) innerFaces.add(f);
            else if (Math.abs(rr - r2v) < 1e-6) outerFaces.add(f);
            else otherFaces.add(f);
            System.out.println("FACE " + f + " r=" + rr + " z=" + c[2]);
        }
        if (innerFaces.isEmpty() || outerFaces.isEmpty())
            throw new IllegalStateException("cylinder faces not identified");
        System.out.println("INNER=" + innerFaces + " OUTER=" + outerFaces
            + " OTHER=" + otherFaces);

        // ---- 材料 ----
        model.component(comp).material().create("mat1", "Common");
        model.component(comp).material("mat1").label("Conductor");
        model.component(comp).material("mat1").selection().set(new int[]{1});
        com.comsol.model.Material mat = model.component(comp).material("mat1");
        mat.propertyGroup("def").set("electricconductivity", new String[][]{{"sig"}});
        mat.propertyGroup("def").set("relpermittivity", new String[][]{{"1"}});
        System.out.println("MAT_OK");

        // ---- 物理场 ----
        model.component(comp).physics().create("ec", "ConductiveMedia", "geom1");
        model.component(comp).physics("ec").create("term1", "Terminal", 2);
        model.component(comp).physics("ec").feature("term1").selection()
            .set(toIntArray(innerFaces));
        model.component(comp).physics("ec").feature("term1").set("TerminalType", "Voltage");
        model.component(comp).physics("ec").feature("term1").set("V0", "V0");
        model.component(comp).physics("ec").create("gnd1", "Ground", 2);
        model.component(comp).physics("ec").feature("gnd1").selection()
            .set(toIntArray(outerFaces));
        System.out.println("PHYS_OK");

        // ---- 网格 ----
        model.component(comp).mesh().create("mesh1");
        com.comsol.model.MeshFeature ftet1 = model.component(comp).mesh("mesh1")
            .create("ftet1", "FreeTet");
        com.comsol.model.MeshFeature size1 = ftet1.create("size1", "Size");
        size1.set("custom", "on");
        size1.set("hmax", "mh");
        size1.set("hmin", "mh/2");
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

        model.save(P[3], "java");
        System.out.println("EcCylinderOrder2Stationary_OK");
    }

    private static int[] toIntArray(java.util.List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) out[i] = values.get(i);
        return out;
    }
}
