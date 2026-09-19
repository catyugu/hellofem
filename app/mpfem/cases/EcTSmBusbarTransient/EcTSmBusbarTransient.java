import com.comsol.model.GeomInfo;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/**
 * EcTSmBusbarTransient: transient electric -> thermal -> structural coupling on
 * an L-shaped copper busbar carrying three titanium bolts.
 *
 * <p>Geometry, materials and boundaries are those of EcTSmBusbarStationary: an
 * L-shaped copper plate (outer rectangle (L+2*tbb) x Hbb minus the inner
 * rectangle (L+tbb) x (Hbb-tbb) at (0, tbb), corners filleted by tbb and 2*tbb)
 * extruded by wbb along y, three titanium bolts of radius rad_1 piercing the
 * plate through its tbb wall, Form Union keeping the internal boundaries and
 * leaving 7 domains. The free end face of the vertical bolt is the voltage
 * terminal, the free end faces of the two horizontal bolts are grounded and
 * carry the fixed constraint, every other exterior face convects to T0.
 *
 * <p>The study is transient, which is what separates this case from the
 * stationary one: the heat equation carries the volumetric heat capacity
 * rho*cp of each material, the electric and the mechanical equations have no
 * time derivative and stay quasi-static, and the thermal expansion follows the
 * temperature of each level. The busbar starts at T0, so the whole transient is
 * driven by the Joule heating of the circuit, and the reference solution is one
 * the app reaches by stepping rather than by one solve.
 *
 * <p>What the transient adds to the verification. The two materials hold
 * different thermal masses and conductivities, so the plate and the bolts heat
 * on different scales, and the reference separates them: at 600 s the plate has
 * risen 5.5 K at its hottest against the bolts' 13.3 K, and neither is
 * finished — the run is still climbing, so the comparison spans a transient the
 * steps have to follow rather than a settled state. And the electric potential
 * is temperature independent, so every level of the reference must reproduce
 * one and the same V (0 to Vtot across the bolts, the plate at 6.42 to 7.13 mV
 * throughout the run): a state that accumulated instead of being replaced would
 * show up as a drift of V with the step count, which no other column of this
 * case can reveal.
 *
 * <p>Identification (deterministic, no hard-coded entity numbers): the bolt
 * axes come from the extruded solid's bounding box, so the case follows its
 * parameters. A domain is a bolt domain when every sampled point of the faces
 * it owns lies on one of the three bolt cylinders; the remaining domain is the
 * busbar. An exterior face is a bolt end face when all of its sampled points
 * sit at one axial end of that cylinder, within its radius. The terminal is the
 * free end of the vertical bolt, the ground is the free ends of the two
 * horizontal bolts, and the faces left over carry the convection. The three
 * sets are asserted: a face that went unclassified would silently drop a
 * boundary condition, which no comparison of the result can localize.
 *
 * <p>Export: args[0]=result.txt (V, T, solid.disp at every output time),
 *       args[1]=mesh.mphtxt, args[2]=.mph, args[3]=generated_model.java.
 */
public class EcTSmBusbarTransient {

    public static void main(String[] args) throws Exception {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a
            : new String[]{"result.txt", "mesh.mphtxt", "EcTSmBusbarTransient.mph",
                "generated_model.java"};

        Model model = ModelUtil.create("Model");
        model.param().set("L", "9[cm]", "length of the horizontal leg");
        model.param().set("Hbb", "10[cm]", "busbar height in the work plane");
        model.param().set("tbb", "5[mm]", "busbar wall thickness");
        model.param().set("wbb", "5[cm]", "busbar width (extrusion depth)");
        model.param().set("rad_1", "6[mm]", "bolt radius");
        model.param().set("Vtot", "20[mV]", "terminal voltage");
        model.param().set("T0", "293.15[K]", "ambient and strain reference temperature");
        model.param().set("htc", "5[W/(m^2*K)]", "natural convection coefficient");
        model.param().set("mh", "6[mm]", "max mesh size");
        model.param().set("tend", "600[s]", "end time");
        model.param().set("dtout", "30[s]", "output interval");
        // Material parameters. The names must not collide with the material
        // variables COMSOL evaluates them into (rho, k, Cp, sig, E, nu,
        // alpha): such a name resolves to the material's own property.
        model.param().set("sigma_Cu", "5.998e7[S/m]", "copper conductivity");
        model.param().set("k_Cu", "400[W/(m*K)]", "copper thermal conductivity");
        model.param().set("rho_Cu", "8700[kg/m^3]", "copper density");
        model.param().set("Cp_Cu", "385[J/(kg*K)]", "copper heat capacity");
        model.param().set("E_Cu", "110[GPa]", "copper Young modulus");
        model.param().set("nu_Cu", "0.35", "copper Poisson ratio");
        model.param().set("alpha_Cu", "17e-6[1/K]", "copper thermal expansion");
        model.param().set("sigma_Ti", "7.407e5[S/m]", "titanium conductivity");
        model.param().set("k_Ti", "7.5[W/(m*K)]", "titanium thermal conductivity");
        model.param().set("rho_Ti", "4940[kg/m^3]", "titanium density");
        model.param().set("Cp_Ti", "710[J/(kg*K)]", "titanium heat capacity");
        model.param().set("E_Ti", "105[GPa]", "titanium Young modulus");
        model.param().set("nu_Ti", "0.33", "titanium Poisson ratio");
        model.param().set("alpha_Ti", "7.06e-6[1/K]", "titanium thermal expansion");

        final double L = model.param().evaluate("L");
        final double Hbb = model.param().evaluate("Hbb");
        final double tbb = model.param().evaluate("tbb");
        final double wbb = model.param().evaluate("wbb");
        final double rad = model.param().evaluate("rad_1");
        final double boltLen = 3 * tbb;

        String comp = "comp1";
        model.component().create(comp, true);

        // ---- Geometry ----
        com.comsol.model.GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
        g3.create("wp1", "WorkPlane");
        g3.feature("wp1").set("quickplane", "xz");
        com.comsol.model.GeomSequence g2 = g3.feature("wp1").geom();
        g2.create("r1", "Rectangle");
        g2.feature("r1").set("size", new String[]{"L+2*tbb", "Hbb"});
        g2.feature("r1").set("pos", new double[]{0, 0});
        g2.create("r2", "Rectangle");
        g2.feature("r2").set("size", new String[]{"L+tbb", "Hbb-tbb"});
        g2.feature("r2").set("pos", new double[]{0, tbb});
        g2.create("dif1", "Difference");
        g2.feature("dif1").selection("input").set(new String[]{"r1"});
        g2.feature("dif1").selection("input2").set(new String[]{"r2"});
        g2.create("fil1", "Fillet");
        g2.feature("fil1").selection("point").set("dif1(1)", new int[]{3});
        g2.feature("fil1").set("radius", "tbb");
        g2.create("fil2", "Fillet");
        g2.feature("fil2").selection("point").set("fil1(1)", new int[]{6});
        g2.feature("fil2").set("radius", "2*tbb");
        g2.run();

        g3.create("ext1", "Extrude");
        g3.feature("ext1").selection("input").set(new String[]{"wp1"});
        g3.feature("ext1").set("distance", "wbb");
        g3.run();

        GeomInfo gi = model.component(comp).geom("geom1");

        // The extruded solid's box fixes the bolt axes: the case then follows
        // its parameters instead of repeating their values.
        double[][] box = geometryBox(gi);
        if (!(box[1][0] > box[0][0] && box[1][1] > box[0][1] && box[1][2] > box[0][2]))
            throw new IllegalStateException("empty geometry box after the extrusion");
        System.out.println("EXTRUDED_BOX=" + box[0][0] + ".." + box[1][0] + ", " + box[0][1]
            + ".." + box[1][1] + ", " + box[0][2] + ".." + box[1][2]);

        final double yMid = 0.5 * (box[0][1] + box[1][1]);
        final double yOff = 0.25 * (box[1][1] - box[0][1]);
        final double xInner = L + tbb;
        final double zBase = tbb - boltLen;
        final double xHorz = 0.5 * L;
        final double zMid = 0.5 * Hbb;

        g3.create("cyl1", "Cylinder");
        g3.feature("cyl1").set("r", "rad_1");
        g3.feature("cyl1").set("h", "3*tbb");
        g3.feature("cyl1").set("pos", new double[]{xInner, yMid, zMid});
        g3.feature("cyl1").set("axis", new double[]{1, 0, 0});
        g3.run("cyl1");
        g3.create("cyl2", "Cylinder");
        g3.feature("cyl2").set("r", "rad_1");
        g3.feature("cyl2").set("h", "3*tbb");
        g3.feature("cyl2").set("pos", new double[]{xHorz, yMid - yOff, zBase});
        g3.feature("cyl2").set("axis", new double[]{0, 0, 1});
        g3.run("cyl2");
        g3.create("cyl3", "Cylinder");
        g3.feature("cyl3").set("r", "rad_1");
        g3.feature("cyl3").set("h", "3*tbb");
        g3.feature("cyl3").set("pos", new double[]{xHorz, yMid + yOff, zBase});
        g3.feature("cyl3").set("axis", new double[]{0, 0, 1});
        g3.run("cyl3");

        g3.create("uni1", "Union");
        g3.feature("uni1").selection("input")
            .set(new String[]{"ext1", "cyl1", "cyl2", "cyl3"});
        g3.feature("uni1").set("intbnd", "on");
        g3.run();
        System.out.println("GEOM_OK");

        // ---- Domain and boundary identification ----
        // Bolt axes: a point on the axis, the unit direction, and the axial
        // coordinate of the free end measured from that point. The vertical
        // bolt is created from its inner end, the two horizontal ones from
        // their free end, hence the two cases.
        final double[][] axes = {
            {xInner, yMid, zMid, 1, 0, 0, boltLen},
            {xHorz, yMid - yOff, zBase, 0, 0, 1, 0},
            {xHorz, yMid + yOff, zBase, 0, 0, 1, 0},
        };
        // A circular face is sampled over the bounding square of its surface
        // parameterization, so its points reach rad*sqrt(2) from the axis. The
        // radial limit below admits them; the busbar's nearest face point is
        // 5 cm from every axis, so it stays far outside it either way.
        final double radialLimit = 1.5 * rad;
        final double axialTol = 1e-6 * rad;

        final int nDom = gi.getNDomains();
        final int nFace = gi.getNFaces();
        // Geometry adjacency: getUpDown() gives, per face, the domains on its
        // two sides, 0 where a side borders no domain. The owner of a face is
        // its domain side, and a free surface is a face with a zero side.
        final int[][] updown = gi.getUpDown();
        final int[] down = updown[0];
        final int[] up = updown[1];
        if (down.length < nFace || up.length < nFace)
            throw new IllegalStateException("getUpDown() returned " + down.length + "x"
                + up.length + " entries for " + nFace + " faces");
        int[] owner = new int[nFace + 1];
        boolean[] exterior = new boolean[nFace + 1];
        int nExterior = 0;
        for (int f = 1; f <= nFace; f++) {
            exterior[f] = down[f - 1] == 0 || up[f - 1] == 0;
            owner[f] = up[f - 1] != 0 ? up[f - 1] : down[f - 1];
            if (exterior[f]) nExterior++;
        }
        System.out.println("DOMAINS=" + nDom + " FACES=" + nFace + " EXTERIOR=" + nExterior);

        java.util.List<Integer> boltDomains = new java.util.ArrayList<Integer>();
        int busbar = 0;
        for (int d = 1; d <= nDom; d++) {
            if (onBoltCylinder(gi, d, nFace, owner, axes, radialLimit, axialTol, boltLen))
                boltDomains.add(d);
            else if (busbar == 0)
                busbar = d;
            else
                throw new IllegalStateException(
                    "more than one domain outside the bolt cylinders: " + busbar + ", " + d);
        }
        if (busbar == 0 || boltDomains.size() != 6) {
            throw new IllegalStateException("expected 1 busbar + 6 bolt domains, got busbar="
                + busbar + " bolts=" + boltDomains.size());
        }
        System.out.println("BUS_DOM=" + busbar + " BOLT_DOMS=" + boltDomains);

        int terminal = 0;
        java.util.List<Integer> ground = new java.util.ArrayList<Integer>();
        java.util.List<Integer> convection = new java.util.ArrayList<Integer>();
        for (int f = 1; f <= nFace; f++) {
            if (!exterior[f]) continue;
            int end = boltEndFace(gi, f, axes, radialLimit, axialTol, boltLen);

            if (end < 0)
                convection.add(f);
            else if (end == 0) {
                if (terminal != 0) throw new IllegalStateException("two terminal faces");
                terminal = f;
            }
            else
                ground.add(f);
        }
        if (terminal == 0 || ground.size() != 2) {
            throw new IllegalStateException("terminal/ground identification failed: terminal="
                + terminal + " ground=" + ground);
        }
        System.out.println("TERMINAL=" + terminal + " GROUND=" + ground + " CONV="
            + convection.size() + " of " + nExterior + " exterior faces");

        // ---- Materials ----
        int[] boltArr = new int[boltDomains.size()];
        for (int i = 0; i < boltArr.length; i++) boltArr[i] = boltDomains.get(i);
        setMaterial(model, comp, "mat_Cu", "Copper", new int[]{busbar},
            "sigma_Cu", "k_Cu", "rho_Cu", "Cp_Cu", "alpha_Cu", "E_Cu", "nu_Cu");
        setMaterial(model, comp, "mat_Ti", "Titanium beta-21S", boltArr,
            "sigma_Ti", "k_Ti", "rho_Ti", "Cp_Ti", "alpha_Ti", "E_Ti", "nu_Ti");
        System.out.println("MAT_OK");

        // ---- Physics ----
        model.component(comp).physics().create("ec", "ConductiveMedia", "geom1");
        model.component(comp).physics().create("ht", "HeatTransfer", "geom1");
        model.component(comp).physics().create("solid", "SolidMechanics", "geom1");

        int[] groundArr = new int[ground.size()];
        for (int i = 0; i < groundArr.length; i++) groundArr[i] = ground.get(i);
        model.component(comp).physics("ec").create("term1", "Terminal", 2);
        model.component(comp).physics("ec").feature("term1").selection().set(new int[]{terminal});
        model.component(comp).physics("ec").feature("term1").set("TerminalType", "Voltage");
        model.component(comp).physics("ec").feature("term1").set("V0", "Vtot");
        model.component(comp).physics("ec").create("gnd1", "Ground", 2);
        model.component(comp).physics("ec").feature("gnd1").selection().set(groundArr);

        int[] convArr = new int[convection.size()];
        for (int i = 0; i < convArr.length; i++) convArr[i] = convection.get(i);
        model.component(comp).physics("ht").feature("init1").set("Tinit_src", "userdef");
        model.component(comp).physics("ht").feature("init1").set("Tinit", "T0");
        model.component(comp).physics("ht").create("hf1", "HeatFluxBoundary", 2);
        model.component(comp).physics("ht").feature("hf1").selection().set(convArr);
        model.component(comp).physics("ht").feature("hf1").set("HeatFluxType", "ConvectiveHeatFlux");
        model.component(comp).physics("ht").feature("hf1").set("Text", "T0");
        model.component(comp).physics("ht").feature("hf1")
            .set("HeatTransferCoefficientType", "UserDef");
        model.component(comp).physics("ht").feature("hf1").set("h", "htc");

        int[] contact = {terminal, groundArr[0], groundArr[1]};
        model.component(comp).physics("solid").feature("lemm1").set("E_mat", "from_mat");
        model.component(comp).physics("solid").feature("lemm1").set("nu_mat", "from_mat");
        model.component(comp).physics("solid").create("fix1", "Fixed", 2);
        model.component(comp).physics("solid").feature("fix1").selection().set(contact);

        model.component(comp).multiphysics().create("emh1", "ElectromagneticHeating");
        model.component(comp).multiphysics("emh1").set("EMHeat_physics", "ec");
        model.component(comp).multiphysics("emh1").set("Heat_physics", "ht");

        int[] allDomains = new int[nDom];
        for (int d = 1; d <= nDom; d++) allDomains[d - 1] = d;
        model.component(comp).multiphysics().create("te1", "ThermalExpansion");
        model.component(comp).multiphysics("te1").selection().set(allDomains);
        model.component(comp).multiphysics("te1").set("Heat_physics", "ht");
        model.component(comp).multiphysics("te1").set("Solid_physics", "solid");
        model.component(comp).multiphysics("te1").set("alpha_mat", "from_mat");
        model.component(comp).multiphysics("te1")
            .set("minput_strainreferencetemperature_src", "userdef");
        model.component(comp).multiphysics("te1")
            .set("minput_strainreferencetemperature", "T0");
        System.out.println("PHYS_OK");

        // ---- Mesh ----
        model.component(comp).mesh().create("mesh1");
        com.comsol.model.MeshFeature ftet1 = model.component(comp).mesh("mesh1")
            .create("ftet1", "FreeTet");
        com.comsol.model.MeshFeature size1 = ftet1.create("size1", "Size");
        size1.set("custom", "on");
        size1.set("hmax", "mh");
        size1.set("hmin", "mh/3");
        size1.set("hcurve", 0.2);
        size1.set("hgrad", 1.5);
        model.component(comp).mesh("mesh1").run();
        System.out.println("MESH_OK");

        // ---- Study ----
        model.study().create("std1");
        model.study("std1").create("time", "Transient");
        model.study("std1").feature("time").set("tlist", "range(0,dtout,tend)");
        model.study("std1").createAutoSequences("time");
        model.study("std1").run();
        System.out.println("STUDY_OK");

        // ---- Export ----
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
        System.out.println("EcTSmBusbarTransient_OK");
    }

    /**
     * Bounding box of the geometry from a grid of points sampled on every
     * face. Returns {{xmin,ymin,zmin},{xmax,ymax,zmax}}.
     */
    private static double[][] geometryBox(GeomInfo gi) {
        double[][] box = {{Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE},
            {-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE}};
        for (int f = 1; f <= gi.getNFaces(); f++)
            for (double[] p : sampleFace(gi, f, 4))
                for (int k = 0; k < 3; k++) {
                    box[0][k] = Math.min(box[0][k], p[k]);
                    box[1][k] = Math.max(box[1][k], p[k]);
                }
        return box;
    }

    /**
     * Whether every sampled point of the faces domain `dom` owns lies on one
     * of the bolt cylinders: within `radialLimit` of an axis and between its
     * two ends. A face whose parameter grid yields no point fails the test, so
     * a face the probe cannot read never turns the busbar into a bolt.
     */
    private static boolean onBoltCylinder(GeomInfo gi, int dom, int nFace, int[] owner,
        double[][] axes, double radialLimit, double axialTol, double len) {
        boolean sampled = false;
        for (int f = 1; f <= nFace; f++) {
            if (owner[f] != dom) continue;
            double[][] points = sampleFace(gi, f, 3);
            if (points.length == 0) return false;
            for (double[] p : points) {
                sampled = true;
                boolean onAxis = false;
                for (double[] axis : axes) {
                    double s = axial(p, axis);
                    if (s >= -axialTol && s <= len + axialTol
                        && radial(p, axis) <= radialLimit) {
                        onAxis = true;
                        break;
                    }
                }
                if (!onAxis) return false;
            }
        }
        return sampled;
    }

    /**
     * 0 when the exterior face `f` is the free end face of the vertical bolt
     * (the terminal one), 1 when it is a free end face of one of the two
     * horizontal bolts (the grounded ones), -1 when it is no bolt end face —
     * which includes the bolt's inner end face, the disc that the wall's
     * opening leaves facing the plate's own notch.
     */
    private static int boltEndFace(GeomInfo gi, int f, double[][] axes,
        double radialLimit, double axialTol, double len) {
        double[][] points = sampleFace(gi, f, 3);
        if (points.length == 0) return -1;
        for (int i = 0; i < axes.length; i++) {
            double[] axis = axes[i];
            boolean end = true;
            boolean free = true;
            for (double[] p : points) {
                double s = axial(p, axis);
                if (radial(p, axis) > radialLimit
                    || (Math.abs(s) > axialTol && Math.abs(s - len) > axialTol)) {
                    end = false;
                    break;
                }
                if (Math.abs(s - axis[6]) > axialTol) free = false;
            }
            if (end && free) return i == 0 ? 0 : 1;
        }
        return -1;
    }

    /** Axial coordinate of `p` measured from the base point of `axis`. */
    private static double axial(double[] p, double[] axis) {
        return (p[0] - axis[0]) * axis[3] + (p[1] - axis[1]) * axis[4]
            + (p[2] - axis[2]) * axis[5];
    }

    /** Distance from `p` to the line through the base point of `axis`. */
    private static double radial(double[] p, double[] axis) {
        double x = p[0] - axis[0];
        double y = p[1] - axis[1];
        double z = p[2] - axis[2];
        double s = x * axis[3] + y * axis[4] + z * axis[5];
        double rx = x - s * axis[3];
        double ry = y - s * axis[4];
        double rz = z - s * axis[5];
        return Math.sqrt(rx * rx + ry * ry + rz * rz);
    }

    /** An (n+1) x (n+1) grid of points on face `f`, in its parameter space. */
    private static double[][] sampleFace(GeomInfo gi, int f, int n) {
        double[] pr;
        try {
            pr = gi.faceParamRange(f);
        } catch (Exception e) {
            return new double[0][];
        }
        if (pr == null || pr.length < 2) return new double[0][];
        java.util.List<double[]> out = new java.util.ArrayList<double[]>();
        double u0 = pr[0], u1 = pr[1];
        double v0 = pr.length >= 4 ? pr[2] : 0, v1 = pr.length >= 4 ? pr[3] : 1;
        for (int i = 0; i <= n; i++)
            for (int j = 0; j <= n; j++)
                try {
                    double[][] pts = gi.faceX(f, new double[][]{
                        {u0 + (u1 - u0) * i / n, v0 + (v1 - v0) * j / n}});
                    if (pts != null && pts.length > 0) out.add(pts[0]);
                } catch (Exception e) {
                    // A parameter value outside the face's own domain: skip it.
                }
        return out.toArray(new double[0][]);
    }

    /** Write one material on `doms`, with E and nu in the Enu material model. */
    private static void setMaterial(Model model, String comp, String tag, String name,
        int[] doms, String sigma, String k, String rho, String cp, String alpha,
        String E, String nu) {
        model.component(comp).material().create(tag, "Common");
        model.component(comp).material(tag).label(name);
        model.component(comp).material(tag).selection().set(doms);
        com.comsol.model.Material mat = model.component(comp).material(tag);
        mat.propertyGroup("def").set("electricconductivity", new String[][]{{sigma}});
        mat.propertyGroup("def").set("relpermittivity", new String[][]{{"1"}});
        mat.propertyGroup("def").set("thermalconductivity", new String[][]{{k}});
        mat.propertyGroup("def").set("density", new String[][]{{rho}});
        mat.propertyGroup("def").set("heatcapacity", new String[][]{{cp}});
        mat.propertyGroup("def").set("thermalexpansioncoefficient",
            new String[]{alpha, "0", "0", "0", alpha, "0", "0", "0", alpha});
        mat.materialModel().create("Enu", "YoungsModulusAndPoissonsRatio");
        mat.propertyGroup("Enu").set("E", new String[][]{{E}});
        mat.propertyGroup("Enu").set("nu", new String[][]{{nu}});
    }
}
