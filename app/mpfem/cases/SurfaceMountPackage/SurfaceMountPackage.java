import com.comsol.model.GeomInfo;
import com.comsol.model.GeomSequence;
import com.comsol.model.Model;
import com.comsol.model.Selection;
import com.comsol.model.util.ModelUtil;

/**
 * SurfaceMountPackage: the Heat Transfer Module's "Heat Transfer in a
 * Surface-Mount Package for a Silicon Chip" — steady conduction in a chip
 * package soldered to a board, whose copper traces and interconnect are
 * modelled as thin layers on the boundaries rather than as domains.
 *
 * <p>Geometry (the sequence is COMSOL's own, in mm): a 20 x 10 x 1 mm PC
 * board with its top face at z = -0.9; a 9.9 x 3.9 x 0.2 mm chip package
 * centred at the origin, capped by a 0.65 mm tapered hexahedron; 16 gull-wing
 * leads, each a lead block revolved 90 degrees, extruded 0.322 mm, revolved
 * back 90 degrees and extruded 0.16 mm, arrayed 8 times at a 1.27 mm pitch and
 * mirrored in y; a 1 x 1 x 0.1 mm chip at the origin, inside the package; a
 * work plane that imprints a 6 x 4 mm pad, split into a 2 mm strip and the
 * rest, on the board's top face; and a work plane whose 4.145 x 2.15 mm
 * rectangle minus a 3.745 x 1.75 mm one imprints a 0.4 mm copper ring on the
 * package's mid-plane. Form Union leaves 19 domains (board, package, chip and
 * the 16 leads), 398 faces of which 358 are exterior and 40 interior.
 *
 * <p>Materials: FR4 on the board, Plastic on the package, Silicon on the chip,
 * Aluminum on the leads, and Copper as a <em>surface</em> material — a
 * boundary (dimension 2) selection — on the two copper-layer faces. The thin
 * layers take their conductivity, density and heat capacity from that material.
 *
 * <p>Physics: a 2e8 W/m3 source in the chip (0.2 W over its 1 mm3), natural
 * convection (h = 50 W/(m2 K) towards 30 degC) on every exterior face except
 * the 2 mm pad strip, which is held at 50 degC, and two thin layers of copper
 * on the copper faces — 100 um on the trace imprinted on the board's surface,
 * 5 um on the interconnect inside the package. Both are thermally thin
 * (COMSOL's "Conductive" layer type): the layer adds its tangential
 * conduction and its own heat capacity to the boundary, with no temperature
 * difference across its thickness.
 *
 * <p>Study: stationary. The reference is the COMSOL solution of this model;
 * the case exists for the surface material and the thin-layer boundary, whose
 * effect on the package temperature is a few kelvin.
 */
public class SurfaceMountPackage {

    public static void main(String[] args) throws Exception {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a
            : new String[]{"result.txt", "mesh.mphtxt", "SurfaceMountPackage.mph",
                           "generated_model.java"};

        Model model = ModelUtil.create("Model");
        model.label("surface_mount_package.mph");
        model.title("Heat Transfer in a Surface-Mount Package for a Silicon Chip");
        model.description("This example investigates the stationary temperature "
            + "distribution in an integrated circuit mounted close to a hot "
            + "component. The example uses the Heat Transfer interface and its "
            + "Thin Layer feature.");

        // Geometry, in mm (the geometry's own length unit).
        model.param().set("Lb", "20[mm]", "board length");
        model.param().set("Wb", "10[mm]", "board width");
        model.param().set("Tb", "1[mm]", "board thickness");
        model.param().set("zb0", "-1.9[mm]", "board bottom");
        model.param().set("Lp", "9.9[mm]", "package length");
        model.param().set("Wp", "3.9[mm]", "package width");
        model.param().set("Tp", "0.2[mm]", "package thickness");
        model.param().set("xpin", "4.95[mm]", "cap half length");
        model.param().set("ypin0", "1.95[mm]", "cap half width at its bottom");
        model.param().set("ypin1", "1.72[mm]", "cap half width at its top");
        model.param().set("zpin0", "0.1[mm]", "cap bottom");
        model.param().set("zpin1", "0.75[mm]", "cap top");
        model.param().set("xlead0", "-4.645[mm]", "lead block x");
        model.param().set("ylead0", "-2.21[mm]", "lead block y");
        model.param().set("zlead0", "-0.1[mm]", "lead block z");
        model.param().set("Llead", "0.4[mm]", "lead block length");
        model.param().set("Wlead", "0.26[mm]", "lead block width");
        model.param().set("Tlead", "0.2[mm]", "lead block thickness");
        model.param().set("pitch", "1.27[mm]", "lead pitch");
        model.param().set("nlead", "8", "leads per row");
        model.param().set("Wpad", "6[mm]", "pad length");
        model.param().set("Hpad", "4[mm]", "pad width");
        model.param().set("xpad0", "-10[mm]", "pad x");
        model.param().set("ypad0", "-5[mm]", "pad y");
        model.param().set("Wstrip", "2[mm]", "pad strip width");
        model.param().set("Linter0", "4.145[mm]", "interconnect outer length");
        model.param().set("Winter0", "2.15[mm]", "interconnect outer width");
        model.param().set("xinter0", "-4.645[mm]", "interconnect outer x");
        model.param().set("yinter0", "-1.95[mm]", "interconnect outer y");
        model.param().set("Linter1", "3.745[mm]", "interconnect inner length");
        model.param().set("Winter1", "1.75[mm]", "interconnect inner width");
        model.param().set("xinter1", "-4.245[mm]", "interconnect inner x");
        model.param().set("Lchip", "1[mm]", "chip length");
        model.param().set("Wchip", "1[mm]", "chip width");
        model.param().set("Tchip", "0.1[mm]", "chip thickness");

        // Physics, in SI units (the geometry unit does not apply to these).
        model.param().set("q_chip", "2e8[W/m^3]", "chip source density");
        model.param().set("hconv", "50[W/(m^2*K)]", "convective coefficient");
        model.param().set("T_amb", "30[degC]", "ambient temperature");
        model.param().set("T_pad", "50[degC]", "pad temperature");
        model.param().set("Tinit", "293.15[K]", "initial temperature");
        model.param().set("lth1", "1e-4[m]", "copper trace thickness");
        model.param().set("lth2", "5e-6[m]", "interconnect thickness");

        String comp = "comp1";
        model.component().create(comp, true);

        GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
        g3.lengthUnit("mm");

        // Board.
        g3.create("blk1", "Block");
        g3.feature("blk1").label("PC Board");
        g3.feature("blk1").set("size", new String[]{"Lb", "Wb", "Tb"});
        g3.feature("blk1").set("pos", new String[]{"-Lb/2", "-Wb/2", "zb0"});
        g3.feature("blk1").set("selresult", true);
        g3.run("blk1");

        // Package body, centred at the origin.
        g3.create("blk2", "Block");
        g3.feature("blk2").set("size", new String[]{"Lp", "Wp", "Tp"});
        g3.feature("blk2").set("base", "center");
        g3.run("blk2");

        // Tapered cap on top of the package: 0.1 to 0.75 mm, its width
        // narrowing from 1.95 to 1.72 mm.
        g3.create("hex1", "Hexahedron");
        g3.feature("hex1").set("p", new String[][]{
            {"-xpin", "xpin", "xpin", "-xpin", "-xpin", "xpin", "xpin", "-xpin"},
            {"-ypin0", "-ypin0", "ypin0", "ypin0", "-ypin1", "-ypin1", "ypin1", "ypin1"},
            {"zpin0", "zpin0", "zpin0", "zpin0", "zpin1", "zpin1", "zpin1", "zpin1"}});
        g3.run("hex1");

        g3.create("mir1", "Mirror");
        g3.feature("mir1").selection("input").set("hex1");
        g3.feature("mir1").set("keep", true);
        g3.run("mir1");

        g3.create("uni1", "Union");
        g3.feature("uni1").label("Chip Package");
        g3.feature("uni1").selection("input").set("blk2", "hex1", "mir1");
        g3.feature("uni1").set("intbnd", false);
        g3.feature("uni1").set("selresult", true);
        g3.run("uni1");

        // One lead: a block at the package's edge, revolved 90 degrees about
        // an x-parallel axis, extruded along that axis, revolved back and
        // extruded again, which bends it down to the board.
        g3.create("blk3", "Block");
        g3.feature("blk3").set("size", new String[]{"Llead", "Wlead", "Tlead"});
        g3.feature("blk3").set("pos", new String[]{"xlead0", "ylead0", "zlead0"});
        g3.run("blk3");

        g3.feature().create("rev1", "Revolve");
        g3.feature("rev1").selection("inputface").set("blk3", 3);
        g3.feature("rev1").set("angtype", "specang");
        g3.feature("rev1").set("angle2", 90);
        g3.feature("rev1").set("axistype", "3d");
        g3.feature("rev1").set("pos3", new double[]{0, -2.211, -0.24});
        g3.feature("rev1").set("axis3", new int[]{1, 0, 0});
        g3.run("rev1");

        g3.feature().create("ext1", "Extrude");
        g3.feature("ext1").selection("inputface").set("rev1", 2);
        g3.feature("ext1").setIndex("distance", 0.322, 0);
        g3.run("ext1");

        g3.feature().create("rev2", "Revolve");
        g3.feature("rev2").selection("inputface").set("ext1", 3);
        g3.feature("rev2").set("angtype", "specang");
        g3.feature("rev2").set("angle2", -90);
        g3.feature("rev2").set("axistype", "3d");
        g3.feature("rev2").set("pos3", new double[]{0, -2.69, -0.561});
        g3.feature("rev2").set("axis3", new int[]{1, 0, 0});
        g3.run("rev2");

        g3.feature().create("ext2", "Extrude");
        g3.feature("ext2").selection("inputface").set("rev2", 2);
        g3.feature("ext2").setIndex("distance", 0.16, 0);
        g3.run("ext2");

        g3.create("uni2", "Union");
        g3.feature("uni2").selection("input").set("ext2");
        g3.feature("uni2").set("intbnd", false);
        g3.run("uni2");

        // A row of leads, collected into the "Pins" cumulative selection, and
        // its mirror image in y: 16 leads.
        g3.create("arr1", "Array");
        g3.feature("arr1").selection("input").set("uni2");
        g3.feature("arr1").set("fullsize", new String[]{"nlead", "1", "1"});
        g3.feature("arr1").set("displ", new String[]{"pitch", "0", "0"});
        g3.selection().create("csel1", "CumulativeSelection");
        g3.selection("csel1").label("Pins");
        g3.feature("arr1").set("contributeto", "csel1");
        g3.run("arr1");

        g3.create("mir2", "Mirror");
        g3.feature("mir2").set("keep", true);
        g3.feature("mir2").selection("input").named("csel1");
        g3.feature("mir2").set("contributeto", "csel1");
        g3.feature("mir2").set("axis", new int[]{0, 1, 0});
        g3.run("mir2");

        // Work plane 1: a 6 x 4 mm pad on the board's top face, split by a
        // 2 mm layer into the strip held at 50 degC and the copper trace.
        g3.create("wp1", "WorkPlane");
        g3.feature("wp1").set("unite", true);
        g3.feature("wp1").set("quickz", "zb0+Tb");
        g3.feature("wp1").geom().create("r1", "Rectangle");
        g3.feature("wp1").geom().feature("r1").set("size", new String[]{"Wpad", "Hpad"});
        g3.feature("wp1").geom().feature("r1").set("pos", new String[]{"xpad0", "ypad0"});
        g3.feature("wp1").geom().feature("r1").setIndex("layername", "Layer 1", 0);
        g3.feature("wp1").geom().feature("r1").set("layer", new String[]{"Wstrip"});
        g3.feature("wp1").geom().feature("r1").set("layerbottom", false);
        g3.feature("wp1").geom().feature("r1").set("layerleft", true);
        g3.run("wp1");

        // Work plane 2: the interconnect, a 0.4 mm copper ring imprinted on
        // the package's mid-plane.
        g3.create("wp2", "WorkPlane");
        g3.feature("wp2").set("unite", true);
        g3.feature("wp2").label("Interconnect");
        g3.feature("wp2").set("selresult", true);
        g3.feature("wp2").geom().create("r1", "Rectangle");
        g3.feature("wp2").geom().feature("r1").set("size", new String[]{"Linter0", "Winter0"});
        g3.feature("wp2").geom().feature("r1").set("pos", new String[]{"xinter0", "yinter0"});
        g3.feature("wp2").geom().run("r1");
        g3.feature("wp2").geom().create("r2", "Rectangle");
        g3.feature("wp2").geom().feature("r2").set("size", new String[]{"Linter1", "Winter1"});
        g3.feature("wp2").geom().feature("r2").set("pos", new String[]{"xinter1", "yinter0"});
        g3.feature("wp2").geom().run("r2");
        g3.feature("wp2").geom().create("dif1", "Difference");
        g3.feature("wp2").geom().feature("dif1").selection("input").set("r1");
        g3.feature("wp2").geom().feature("dif1").selection("input2").set("r2");
        g3.run("wp2");

        // The chip, centred at the origin inside the package.
        g3.create("blk4", "Block");
        g3.feature("blk4").label("Chip");
        g3.feature("blk4").set("size", new String[]{"Lchip", "Wchip", "Tchip"});
        g3.feature("blk4").set("selresult", true);
        g3.feature("blk4").set("base", "center");
        g3.run("fin");

        // Selections over the finished geometry: everything, its exterior
        // boundaries, and the two copper-layer faces the work planes left.
        g3.create("sel1", "ExplicitSelection");
        g3.feature("sel1").label("Geometry");
        g3.feature("sel1").selection("selection").init();
        g3.feature("sel1").selection("selection").set("fin");
        g3.run("sel1");

        g3.create("adjsel1", "AdjacentSelection");
        g3.feature("adjsel1").label("Exterior Boundaries");
        g3.feature("adjsel1").set("input", new String[]{"sel1"});
        g3.run("adjsel1");

        g3.create("sel2", "ExplicitSelection");
        g3.feature("sel2").selection("selection").init(2);
        g3.feature("sel2").selection("selection").set("fin", 7, 37);
        g3.feature("sel2").label("Copper Layers");

        GeomInfo gi = g3;
        // The geometry is in mm; a parameter evaluates in SI, so the
        // predicates below work in the geometry's own unit.
        final double mm = model.param().evaluate("1[mm]");
        int nDom = gi.getNDomains();
        int nFace = gi.getNFaces();
        int[][] updown = gi.getUpDown();
        int[] board = entities(model, comp, "geom1_blk1_dom");
        int[] packageDomains = entities(model, comp, "geom1_uni1_dom");
        int[] chip = entities(model, comp, "geom1_blk4_dom");
        int[] leads = entities(model, comp, "geom1_csel1_dom");
        int[] exterior = entities(model, comp, "geom1_adjsel1");
        int[] copper = entities(model, comp, "geom1_sel2");
        System.out.println("GEOM nDom=" + nDom + " nFace=" + nFace
            + " board=" + board[0] + " package=" + packageDomains.length
            + " chip=" + chip[0] + " leads=" + leads.length
            + " exterior=" + exterior.length + " copper=" + copper.length);
        if (board.length != 1 || packageDomains.length != 2 || chip.length != 1
            || leads.length != 16 || exterior.length != 358 || nFace != 398
            || nDom != 19)
            throw new IllegalStateException("unexpected geometry classification");

        // The copper faces: the trace the work plane imprinted on the board's
        // surface (exterior) and the interconnect inside the package
        // (interior, both of its sides the same domain).
        int traceFace = 0;
        int interconnectFace = 0;
        for (int f : copper) {
            if (updown[0][f - 1] == 0 || updown[1][f - 1] == 0)
                traceFace = f;
            else if (updown[0][f - 1] == updown[1][f - 1])
                interconnectFace = f;
        }
        if (traceFace == 0 || interconnectFace == 0)
            throw new IllegalStateException("copper layers not classified");

        // The pad strip the work plane split off the board's top face is the
        // boundary held at 50 degC.
        final double zTop = model.param().evaluate("zb0+Tb") / mm;
        final double padX0 = model.param().evaluate("xpad0") / mm;
        final double padY0 = model.param().evaluate("ypad0") / mm;
        final double padW = model.param().evaluate("Wstrip") / mm;
        final double padH = model.param().evaluate("Hpad") / mm;
        int padFace = 0;
        for (int f = 1; f <= nFace; f++) {
            if (updown[0][f - 1] != 0 && updown[1][f - 1] != 0)
                continue; // interior
            if (updown[0][f - 1] != board[0] && updown[1][f - 1] != board[0])
                continue;
            double[] pr = gi.faceParamRange(f);
            double[][] pts = gi.faceX(f, new double[][]{{(pr[0] + pr[1]) / 2,
                pr.length >= 4 ? (pr[2] + pr[3]) / 2 : 0.5}});
            double cx = pts[0][0];
            double cy = pts[0][1];
            double cz = pts[0][2];
            if (Math.abs(cz - zTop) > 1e-6)
                continue;
            if (cx >= padX0 && cx <= padX0 + padW && cy >= padY0 && cy <= padY0 + padH)
                padFace = f;
        }
        if (padFace == 0)
            throw new IllegalStateException("pad face not found");
        System.out.println("FACES pad=" + padFace + " trace=" + traceFace
            + " interconnect=" + interconnectFace);

        // Materials. The surface material (mat5) is selected on boundaries,
        // not domains, and is what the thin layers read their properties from.
        material(model, comp, "mat1", "Aluminum", leads, 3,
            new String[]{"238[W/(m*K)]", "2700[kg/m^3]", "900[J/(kg*K)]"});
        material(model, comp, "mat2", "FR4 (Circuit Board)", board, 3,
            new String[]{"0.3[W/(m*K)]", "1900[kg/m^3]", "1369[J/(kg*K)]"});
        material(model, comp, "mat3", "Plastic", packageDomains, 3,
            new String[]{"0.2[W/(m*K)]", "2700[kg/m^3]", "900[J/(kg*K)]"});
        material(model, comp, "mat4", "Silicon", chip, 3,
            new String[]{"130[W/(m*K)]", "2329[kg/m^3]", "700[J/(kg*K)]"});
        material(model, comp, "mat5", "Copper", copper, 2,
            new String[]{"400[W/(m*K)]", "8960[kg/m^3]", "385[J/(kg*K)]"});
        System.out.println("MAT_OK");

        // Physics: heat transfer in solids.
        model.component(comp).physics().create("ht", "HeatTransfer", "geom1");
        model.component(comp).physics("ht").feature("init1").set("Tinit_src", "userdef");
        model.component(comp).physics("ht").feature("init1").set("Tinit", "Tinit");

        model.component(comp).physics("ht").create("hs1", "HeatSource", 3);
        model.component(comp).physics("ht").feature("hs1").selection().set(chip);
        model.component(comp).physics("ht").feature("hs1").set("Q0", "q_chip");

        // Convection on every exterior face the temperature node does not
        // hold: the pad strip is the one boundary that is not convecting.
        int[] convecting = new int[exterior.length - 1];
        int n = 0;
        for (int f : exterior)
            if (f != padFace)
                convecting[n++] = f;
        if (n != convecting.length)
            throw new IllegalStateException("pad face is not an exterior face");
        model.component(comp).physics("ht").create("hf1", "HeatFluxBoundary", 2);
        model.component(comp).physics("ht").feature("hf1").selection().set(convecting);
        model.component(comp).physics("ht").feature("hf1").set("HeatFluxType", "ConvectiveHeatFlux");
        model.component(comp).physics("ht").feature("hf1").set("h", "hconv");
        model.component(comp).physics("ht").feature("hf1").set("Text", "T_amb");

        model.component(comp).physics("ht").create("temp1", "TemperatureBoundary", 2);
        model.component(comp).physics("ht").feature("temp1").selection().set(new int[]{padFace});
        model.component(comp).physics("ht").feature("temp1").set("T0", "T_pad");

        thinLayer(model, comp, "sls1", traceFace, "lth1");
        thinLayer(model, comp, "sls2", interconnectFace, "lth2");
        System.out.println("PHYS_OK");

        // Mesh: a finer mesh on the pad and the trace, the default elsewhere.
        model.component(comp).mesh().create("mesh1");
        model.component(comp).mesh("mesh1").create("size1", "Size");
        model.component(comp).mesh("mesh1").feature("size1").selection().geom("geom1", 2);
        model.component(comp).mesh("mesh1").feature("size1").selection()
            .set(new int[]{padFace, traceFace});
        model.component(comp).mesh("mesh1").feature("size1").set("hauto", 2);
        model.component(comp).mesh("mesh1").create("ftet1", "FreeTet");
        model.component(comp).mesh("mesh1").feature("size").set("hauto", 4);
        model.component(comp).mesh("mesh1").run();
        System.out.println("MESH_OK");

        model.study().create("std1");
        model.study("std1").create("stat", "Stationary");
        model.study("std1").feature("stat").setSolveFor("/physics/ht", true);
        model.study("std1").run();
        System.out.println("STUDY_OK");

        model.result().export().create("data1", "Data");
        model.result().export("data1").set("data", "dset1");
        model.result().export("data1").set("filename", P[0]);
        model.result().export("data1").set("expr", new String[]{"T"});
        model.result().export("data1").run();

        model.result().export().create("mesh1", "Mesh");
        model.result().export("mesh1").set("data", "dset1");
        model.result().export("mesh1").set("filename", P[1]);
        model.result().export("mesh1").run();

        model.save(P[3], "java");
        System.out.println("SurfaceMountPackage_OK");
    }

    /// The entities of a geometry selection the sequence created, by the name
    /// the component exposes it under.
    private static int[] entities(Model model, String comp, String tag) {
        Selection s = model.component(comp).selection(tag);
        return s.entities();
    }

    /// A material on a domain (dim 3) or boundary (dim 2) selection, with its
    /// thermal conductivity, density and heat capacity.
    private static void material(Model model, String comp, String tag, String label,
        int[] selection, int dim, String[] properties) {
        model.component(comp).material().create(tag, "Common");
        model.component(comp).material(tag).label(label);
        model.component(comp).material(tag).selection().geom("geom1", dim);
        model.component(comp).material(tag).selection().set(selection);
        String[] names = {"thermalconductivity", "density", "heatcapacity"};
        for (int i = 0; i < names.length; i++)
            model.component(comp).material(tag).propertyGroup("def")
                .set(names[i], new String[][]{{properties[i]}});
    }

    /// A thin layer of the boundary's material: a copper layer of the given
    /// thickness, thermally thin, so that no temperature difference builds up
    /// across it and only its tangential conduction acts.
    private static void thinLayer(Model model, String comp, String tag,
        int boundary, String thickness) {
        model.component(comp).physics("ht").create(tag, "SolidLayeredShell", 2);
        model.component(comp).physics("ht").feature(tag).selection().set(new int[]{boundary});
        model.component(comp).physics("ht").feature(tag).set("lth_mat", "userdef");
        model.component(comp).physics("ht").feature(tag).set("lth", thickness);
        model.component(comp).physics("ht").feature(tag)
            .set("UserDefThicknessLayerType", "Conductive");
        model.component(comp).physics("ht").feature(tag).set("k_mat", "from_mat");
        model.component(comp).physics("ht").feature(tag).set("rho_mat", "from_mat");
        model.component(comp).physics("ht").feature(tag).set("Cp_mat", "from_mat");
    }
}
