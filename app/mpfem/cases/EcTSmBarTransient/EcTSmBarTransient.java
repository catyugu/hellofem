import com.comsol.model.GeomInfo;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/**
 * EcTSmBarTransient: transient electric -> thermal -> structural coupling on a
 * copper busbar.
 *
 * Geometry: 3D Block (L x wbb x tbb), single domain.
 * Physics: ec (Terminal V=V0 on the x=L face, Ground on x=0),
 *   ht (both end faces held at T0, natural convection htc on the four side
 *   faces), solid (both end faces fixed).
 * Coupling: ElectromagneticHeating (ec -> ht) and ThermalExpansion
 *   (ht -> solid, reference temperature T0).
 * Study: Transient, tlist = range(0, dtout, tend). The busbar starts at T0,
 * so the whole transient is driven by the Joule heating; the electric and
 * mechanical equations have no time derivative (the mechanical inertia is
 * negligible at this time scale, ~1e-5 s against 60 s), so they are
 * quasi-static at every time level.
 *
 * The material parameters carry a suffix because a time-dependent study
 * evaluates the density and the conductivity: a parameter named after the
 * material's own variable (rho, k, Cp, sig) would be a self-reference and
 * COMSOL stops with "Undefined variable comp1.mat1.def.rho".
 *
 * Face identification (deterministic): sample each face centre with
 * GeomInfo.faceX and classify by coordinate plane: x~0 -> ground/fixed,
 * x~L -> terminal/fixed, the rest -> convection.
 *
 * Export: args[0]=result.txt (V, T, solid.disp per time level),
 *   args[1]=mesh.mphtxt, args[2]=.mph, args[3]=generated_model.java.
 */
public class EcTSmBarTransient {

    public static void main(String[] args) throws Exception {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a
            : new String[]{"result.txt", "mesh.mphtxt", "EcTSmBarTransient.mph", "generated_model.java"};

        Model model = ModelUtil.create("Model");
        model.param().set("L", "0.1[m]", "busbar length");
        model.param().set("wbb", "0.03[m]", "busbar width");
        model.param().set("tbb", "0.005[m]", "busbar thickness");
        model.param().set("V0", "0.05[V]", "terminal voltage");
        model.param().set("T0", "293.15[K]", "ambient and strain reference temperature");
        model.param().set("htc", "5[W/(m^2*K)]", "natural convection coefficient");
        model.param().set("tend", "60[s]", "end time");
        model.param().set("dtout", "2[s]", "output interval");
        model.param().set("mh", "0.002[m]", "max mesh size");
        // Copper
        model.param().set("sig_Cu", "5.998e7[S/m]", "copper conductivity");
        model.param().set("k_Cu", "400[W/(m*K)]", "copper thermal conductivity");
        model.param().set("rho_Cu", "8700[kg/m^3]", "copper density");
        model.param().set("cp_Cu", "385[J/(kg*K)]", "copper heat capacity");
        model.param().set("E_Cu", "110[GPa]", "copper Young modulus");
        model.param().set("nu_Cu", "0.35", "copper Poisson ratio");
        model.param().set("alpha_Cu", "17e-6[1/K]", "copper thermal expansion");

        String comp = "comp1";
        model.component().create(comp, true);

        com.comsol.model.GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
        g3.create("blk1", "Block");
        g3.feature("blk1").set("size", new String[]{"L", "wbb", "tbb"});
        g3.feature("blk1").set("pos", new String[]{"0", "0", "0"});
        g3.run();
        System.out.println("GEOM_OK");

        GeomInfo gi = model.component(comp).geom("geom1");
        int nFace = gi.getNFaces();
        double Lval = model.param().evaluate("L");
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
            else if (Math.abs(c[0] - Lval) < 1e-6) terminalFaces.add(f);
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

        model.component(comp).material().create("mat1", "Common");
        model.component(comp).material("mat1").label("Copper");
        model.component(comp).material("mat1").selection().set(new int[]{1});
        com.comsol.model.Material mat = model.component(comp).material("mat1");
        mat.propertyGroup("def").set("electricconductivity", new String[][]{{"sig_Cu"}});
        mat.propertyGroup("def").set("relpermittivity", new String[][]{{"1"}});
        mat.propertyGroup("def").set("thermalconductivity", new String[][]{{"k_Cu"}});
        mat.propertyGroup("def").set("density", new String[][]{{"rho_Cu"}});
        mat.propertyGroup("def").set("heatcapacity", new String[][]{{"cp_Cu"}});
        mat.propertyGroup("def").set("thermalexpansioncoefficient",
            new String[]{"alpha_Cu", "0", "0", "0", "alpha_Cu", "0", "0", "0", "alpha_Cu"});
        mat.materialModel().create("Enu", "YoungsModulusAndPoissonsRatio");
        mat.propertyGroup("Enu").set("E", new String[][]{{"E_Cu"}});
        mat.propertyGroup("Enu").set("nu", new String[][]{{"nu_Cu"}});
        System.out.println("MAT_OK");

        model.component(comp).physics().create("ec", "ConductiveMedia", "geom1");
        model.component(comp).physics().create("ht", "HeatTransfer", "geom1");
        model.component(comp).physics().create("solid", "SolidMechanics", "geom1");

        // Electric boundaries
        model.component(comp).physics("ec").create("term1", "Terminal", 2);
        model.component(comp).physics("ec").feature("term1").selection().set(new int[]{terminalFace});
        model.component(comp).physics("ec").feature("term1").set("TerminalType", "Voltage");
        model.component(comp).physics("ec").feature("term1").set("V0", "V0");
        model.component(comp).physics("ec").create("gnd1", "Ground", 2);
        model.component(comp).physics("ec").feature("gnd1").selection().set(new int[]{groundFace});

        // Heat: initial temperature, convection on the sides, end faces at T0
        model.component(comp).physics("ht").feature("init1").set("Tinit_src", "userdef");
        model.component(comp).physics("ht").feature("init1").set("Tinit", "T0");
        model.component(comp).physics("ht").create("hf1", "HeatFluxBoundary", 2);
        model.component(comp).physics("ht").feature("hf1").selection().set(convArr);
        model.component(comp).physics("ht").feature("hf1").set("HeatFluxType", "ConvectiveHeatFlux");
        model.component(comp).physics("ht").feature("hf1").set("minput_temperature_src", "userdef");
        model.component(comp).physics("ht").feature("hf1").set("minput_temperature", "T0");
        model.component(comp).physics("ht").feature("hf1").set("HeatTransferCoefficientType", "UserDef");
        model.component(comp).physics("ht").feature("hf1").set("h", "htc");
        model.component(comp).physics("ht").create("temp1", "TemperatureBoundary", 2);
        model.component(comp).physics("ht").feature("temp1").selection().set(new int[]{groundFace});
        model.component(comp).physics("ht").feature("temp1").set("T0_src", "userdef");
        model.component(comp).physics("ht").feature("temp1").set("T0", "T0");
        model.component(comp).physics("ht").create("temp2", "TemperatureBoundary", 2);
        model.component(comp).physics("ht").feature("temp2").selection().set(new int[]{terminalFace});
        model.component(comp).physics("ht").feature("temp2").set("T0_src", "userdef");
        model.component(comp).physics("ht").feature("temp2").set("T0", "T0");

        // Joule heating
        model.component(comp).multiphysics().create("emh1", "ElectromagneticHeating");
        model.component(comp).multiphysics("emh1").set("EMHeat_physics", "ec");
        model.component(comp).multiphysics("emh1").set("Heat_physics", "ht");

        // Structure: both end faces clamped
        model.component(comp).physics("solid").feature("lemm1").set("E_mat", "from_mat");
        model.component(comp).physics("solid").feature("lemm1").set("nu_mat", "from_mat");
        model.component(comp).physics("solid").create("fix1", "Fixed", 2);
        model.component(comp).physics("solid").feature("fix1").selection().set(new int[]{groundFace});
        model.component(comp).physics("solid").create("fix2", "Fixed", 2);
        model.component(comp).physics("solid").feature("fix2").selection().set(new int[]{terminalFace});

        model.component(comp).multiphysics().create("te1", "ThermalExpansion");
        model.component(comp).multiphysics("te1").selection().set(new int[]{1});
        model.component(comp).multiphysics("te1").set("Heat_physics", "ht");
        model.component(comp).multiphysics("te1").set("Solid_physics", "solid");
        model.component(comp).multiphysics("te1").set("alpha_mat", "from_mat");
        model.component(comp).multiphysics("te1").set("minput_strainreferencetemperature_src", "userdef");
        model.component(comp).multiphysics("te1").set("minput_strainreferencetemperature", "T0");
        System.out.println("PHYS_OK");

        model.component(comp).mesh().create("mesh1");
        com.comsol.model.MeshFeature ftet1 = model.component(comp).mesh("mesh1").create("ftet1", "FreeTet");
        com.comsol.model.MeshFeature size1 = ftet1.create("size1", "Size");
        size1.set("custom", "on");
        size1.set("hmax", "mh");
        size1.set("hmin", "mh/2");
        model.component(comp).mesh("mesh1").run();
        System.out.println("MESH_OK");

        model.study().create("std1");
        model.study("std1").create("time", "Transient");
        model.study("std1").feature("time").set("tlist", "range(0,dtout,tend)");
        model.study("std1").createAutoSequences("time");
        model.study("std1").run();
        System.out.println("STUDY_OK");

        model.result().export().create("data1", "Data");
        model.result().export("data1").set("data", "dset1");
        model.result().export("data1").set("filename", P[0]);
        model.result().export("data1").set("expr", new String[]{"V", "T", "solid.disp"});
        model.result().export("data1").run();

        model.result().export().create("mesh1", "Mesh");
        model.result().export("mesh1").set("data", "dset1");
        model.result().export("mesh1").set("filename", P[1]);
        model.result().export("mesh1").run();

        model.save(P[3], "java");
        System.out.println("EcTSmBarTransient_OK");
    }
}
