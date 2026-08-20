import com.comsol.model.GeomInfo;
import com.comsol.model.Model;
import com.comsol.model.MeshFeature;
import com.comsol.model.util.ModelUtil;

/**
 * NonlinThermal: temperature-dependent thermal conductivity (pure HT, nonlinear).
 *
 * Geometry: 3D Block (0.1 x 0.05 x 0.01 m), single domain.
 * Physics: ht (left T=300K, right T=400K, others insulated).
 * Material: k(T) = k0 * (1 + alpha_k * (T - Tref)).
 * Mesh: FreeTet hmax=mh.
 * Study: Stationary (COMSOL handles nonlinearity natively).
 */
public class NonlinThermal {

    public static void main(String[] args) throws Exception {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a
            : new String[]{"result.txt", "mesh.mphtxt", "NonlinThermal.mph", "generated_model.java"};

        Model model = ModelUtil.create("Model");
        model.param().set("L", "0.1[m]", "bar length");
        model.param().set("W", "0.05[m]", "bar width");
        model.param().set("T", "0.01[m]", "bar thickness");
        model.param().set("Tleft", "300[K]", "left temperature");
        model.param().set("Tright", "400[K]", "right temperature");
        model.param().set("k0", "50[W/(m*K)]", "base conductivity");
        model.param().set("alpha_k", "0.05[1/K]", "conductivity temp coefficient");
        model.param().set("Tref", "293.15[K]", "reference temperature");
        model.param().set("mh", "0.005[m]", "max mesh size");

        String comp = "comp1";
        model.component().create(comp, true);

        // Geometry: rectangular block
        com.comsol.model.GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
        g3.create("blk1", "Block");
        g3.feature("blk1").set("size", new String[]{"L", "W", "T"});
        g3.feature("blk1").set("pos", new String[]{"0", "0", "0"});
        g3.run();
        System.out.println("GEOM_OK");

        // Face identification
        GeomInfo gi = model.component(comp).geom("geom1");
        int nFace = gi.getNFaces();
        java.util.List<Integer> leftFaces = new java.util.ArrayList<Integer>();
        java.util.List<Integer> rightFaces = new java.util.ArrayList<Integer>();
        for (int f = 1; f <= nFace; f++) {
            double[] pr;
            try { pr = gi.faceParamRange(f); } catch (Exception e) { continue; }
            double[][] pts;
            try {
                pts = gi.faceX(f, new double[][]{{(pr[0]+pr[1])/2,
                    pr.length >= 4 ? (pr[2]+pr[3])/2 : 0.5}});
            } catch (Exception e2) {
                try { pts = gi.faceX(f, new double[][]{{0.5, 0.5}}); }
                catch (Exception e3) { continue; }
            }
            if (pts == null || pts.length == 0) continue;
            double[] c = pts[0];
            double Lval = model.param().evaluate("L");
            if (Math.abs(c[0]) < 1e-6) leftFaces.add(f);
            else if (Math.abs(c[0] - Lval) < 1e-6) rightFaces.add(f);
        }
        int leftFace = leftFaces.get(0);
        int rightFace = rightFaces.get(0);
        System.out.println("LEFT=" + leftFace + " RIGHT=" + rightFace);

        // Material: k(T) = k0 * (1 + alpha_k * (T - Tref)), isotropic scalar
        model.component(comp).material().create("mat1", "Common");
        model.component(comp).material("mat1").label("NonlinMat");
        model.component(comp).material("mat1").selection().set(new int[]{1});
        model.component(comp).material("mat1").propertyGroup("def")
            .set("thermalconductivity", new String[][]{{"k0*(1+alpha_k*(T-Tref))"}});
        model.component(comp).material("mat1").propertyGroup("def")
            .set("density", new String[][]{{"1[kg/m^3]"}});
        model.component(comp).material("mat1").propertyGroup("def")
            .set("heatcapacity", new String[][]{{"1[J/(kg*K)]"}});
        System.out.println("MAT_OK");

        // Physics: Heat Transfer
        model.component(comp).physics().create("ht", "HeatTransfer", "geom1");

        // Left boundary: T = Tleft (TemperatureBoundary — COMSOL 6.2 API)
        model.component(comp).physics("ht").create("temp1", "TemperatureBoundary", 2);
        model.component(comp).physics("ht").feature("temp1").selection().set(new int[]{leftFace});
        model.component(comp).physics("ht").feature("temp1").set("T0_src", "userdef");
        model.component(comp).physics("ht").feature("temp1").set("T0", "Tleft");

        // Right boundary: T = Tright
        model.component(comp).physics("ht").create("temp2", "TemperatureBoundary", 2);
        model.component(comp).physics("ht").feature("temp2").selection().set(new int[]{rightFace});
        model.component(comp).physics("ht").feature("temp2").set("T0_src", "userdef");
        model.component(comp).physics("ht").feature("temp2").set("T0", "Tright");

        System.out.println("PHYS_OK");

        // Mesh
        model.component(comp).mesh().create("mesh1");
        MeshFeature ftet1 = model.component(comp).mesh("mesh1").create("ftet1", "FreeTet");
        MeshFeature size1 = ftet1.create("size1", "Size");
        size1.set("custom", "on");
        size1.set("hmax", "mh");
        size1.set("hmin", "mh/2");
        model.component(comp).mesh("mesh1").run();
        System.out.println("MESH_OK");

        // Study: Stationary
        model.study().create("std1");
        model.study("std1").create("stat", "Stationary");
        model.study("std1").createAutoSequences("stat");
        model.study("std1").run();
        System.out.println("STUDY_OK");

        // Export: Data (T only) + Mesh
        model.result().export().create("data1", "Data");
        model.result().export("data1").set("data", "dset1");
        model.result().export("data1").set("filename", P[0]);
        model.result().export("data1").set("expr", new String[]{"T"});
        model.result().export("data1").run();

        model.result().export().create("mesh1", "Mesh");
        model.result().export("mesh1").set("data", "dset1");
        model.result().export("mesh1").set("filename", P[1]);
        model.result().export("mesh1").run();

        model.save(P[2]);
        model.save(P[3], "java");
        System.out.println("NonlinThermal_OK");
    }
}