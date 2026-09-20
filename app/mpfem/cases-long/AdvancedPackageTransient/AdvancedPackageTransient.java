import com.comsol.model.GeomInfo;
import com.comsol.model.GeomSequence;
import com.comsol.model.Model;
import com.comsol.model.Selection;
import com.comsol.model.util.ModelUtil;

/**
 * AdvancedPackageTransient: transient electric -> thermal -> structural
 * coupling in a flip-chip package soldered to a printed circuit board.
 *
 * <p>Geometry (the geometry's own length unit, mm): a 16 x 16 x 1.6 mm FR4
 * board with its top face at z = 0; a 9 x 9 x 0.1 mm copper substrate layer on
 * it; a 7 x 7 x 0.34 mm underfill whose 3 x 3 array of 0.25 mm radius, 0.34 mm
 * tall solder bumps (1.1 mm pitch) is subtracted from it, so the bumps are
 * domains of their own inside the underfill; a 7 x 7 x 0.1 mm copper layer on
 * top of the underfill and the bumps — the die's front-side metal; and a
 * 6 x 6 x 0.74 mm silicon die centred on it. Form Union leaves 14 domains
 * (board, substrate copper, underfill, 9 bumps, die metal, die).
 *
 * <p>Physics. Conductive media, heat transfer and solid mechanics, coupled by
 * electromagnetic heating (Joule heat in the copper layers and the bumps) and
 * thermal expansion (the die's and the board's strain reference is the initial
 * temperature). The terminal is the peripheral ring the die leaves exposed on
 * the die metal, the ground is the ring the underfill leaves exposed on the
 * substrate copper, and every other exterior face — the board's underside
 * included — convects to the initial temperature: the package is cooled by
 * natural convection alone. The die carries a volumetric source of its own
 * (chip power, 3e7 W/m3 over 26.64 mm3 = 0.80 W, 2.2 W/cm2), and the current
 * adds its Joule heat through the same path (2 mV over the package's 37.9 uohm
 * = 0.11 W over 9 bumps). The two are generated in different places and leave
 * through different paths — measured on this model, the bumps' heat through
 * 37.6 K/W and the die's own through 79.1 K/W — so the junction they drive
 * together (86 degC) is set by the bump array rather than by the board. The
 * board's underside is also the mechanical support, so the displacement is
 * fixed there.
 *
 * <p>What the case adds to the verification. The current enters the package on
 * a ring, spreads through the die metal, splits over the 9 bumps in parallel
 * and leaves on the other ring: the solution is a parallel network whose value
 * is fixed by the geometry and the conductivities, so a boundary or a material
 * the app binds to the wrong entity moves the total resistance — and with it V
 * and every temperature — by a factor no discretization error can produce. The
 * five materials put silicon (2.6e-6 1/K), copper (17e-6), FR4 (17e-6),
 * solder (21e-6) and epoxy (30e-6) on one mesh with Young moduli from 8 GPa to
 * 130 GPa and thermal conductivities from 0.3 to 400 W/(m*K): the temperature
 * field is set by the FR4's poor spreading and the displacement by the
 * mismatch between the die and the board, so a property the app reads from the
 * wrong material is visible in both.
 *
 * <p>Identification (deterministic, no hard-coded entity numbers): the
 * geometry features carry their own selections (`geom1_blk*_dom`,
 * `geom1_csel1_dom` for the bumps), and the underfill is the one domain of
 * the block the bumps were subtracted from that is not a bump. The boundaries
 * are found by sampling: an exterior face owned by the die metal whose points
 * all sit at its top face is the terminal, the same on the substrate copper is
 * the ground, and the same on the board's bottom is the fixed one. Each set is
 * asserted — the bump count and the domain count follow `nb`, and the terminal
 * and the ground have to reach all four sides of the ring they sit on —
 * because a face that goes unclassified silently drops a boundary condition,
 * which no comparison of the result can localize.
 *
 * <p>The whole model is written in terms of its parameters, so a larger
 * instance of the same package is a parameter change and not a second model
 * (see `.cache/perf/make_perf_case.py` for the large-scale performance run).
 *
 * <p>Export: args[0]=result.txt (V, T, solid.disp at every output time),
 *       args[1]=mesh.mphtxt, args[2]=.mph, args[3]=generated_model.java.
 */
public class AdvancedPackageTransient {

    public static void main(String[] args) throws Exception {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a
            : new String[]{"result.txt", "mesh.mphtxt", "AdvancedPackageTransient.mph",
                "generated_model.java"};

        Model model = ModelUtil.create("Model");
        model.label("advanced_package.mph");
        model.title("Electro-thermal-structural transient of a flip-chip package");

        // ---- Geometry, in mm (the geometry's own length unit) ----
        model.param().set("Lb", "16[mm]", "board side");
        model.param().set("Tb", "1.6[mm]", "board thickness");
        model.param().set("Ls", "9[mm]", "substrate copper side");
        model.param().set("Ts", "0.1[mm]", "substrate copper thickness");
        model.param().set("Lu", "7[mm]", "underfill side");
        model.param().set("Tu", "0.34[mm]", "underfill thickness");
        model.param().set("rb", "0.25[mm]", "bump radius");
        model.param().set("pitch", "1.1[mm]", "bump pitch");
        model.param().set("nb", "3", "bumps per row");
        model.param().set("Tm", "0.1[mm]", "die metal thickness");
        model.param().set("Ld", "6[mm]", "die side");
        model.param().set("Td", "0.74[mm]", "die thickness");

        // ---- Physics, in SI units (the geometry unit does not apply) ----
        // The operating point. The package's current path is 37.9 uohm (die
        // metal 4.1, the nine bumps in parallel 27.1, substrate 6.7), so the
        // terminal voltage sets the current: 2 mV drives 52.8 A, i.e. 5.9 A
        // through each bump, and the Joule term is 0.106 W against the die's
        // own 0.27 W. A larger voltage would put the package far above the
        // current a 3 x 3 array carries: 8 mV would drive 211 A and 23.5 A per
        // bump, and the Joule heat alone (1.69 W) would take the junction to
        // 105 degC.
        model.param().set("Vtot", "2[mV]", "terminal voltage");
        // The die's own power: 3e7 W/m3 over its 26.64 mm3 is 0.80 W, i.e.
        // 2.2 W/cm2 over its 36 mm2 face — the power density of a chip a
        // package's thermal design is sized for. The two heat inputs see very
        // different paths out of the assembly, measured on this model: the
        // current's Joule heat is generated in the bumps, right above the
        // board, and leaves through 37.6 K/W, while the die's own power has to
        // cross the die, the die metal, the bump array and the board and
        // leaves through 79.1 K/W. Together they put the junction at 86 degC:
        // warm enough that the thermal path is the binding constraint, and
        // below the 125 degC a silicon junction is limited to.
        model.param().set("q_chip", "3e7[W/m^3]", "chip power density");
        model.param().set("hconv", "50[W/(m^2*K)]", "convective coefficient");
        model.param().set("T0", "293.15[K]", "ambient and strain reference temperature");
        model.param().set("tend", "100[s]", "end time");
        model.param().set("dtout", "10[s]", "output interval");
        // Material parameters. The names must not collide with the material
        // variables COMSOL evaluates them into (sigma, k, rho, Cp, E, nu,
        // alpha): such a name resolves to the material's own property.
        model.param().set("sigma_Cu", "5.998e7[S/m]", "copper conductivity");
        model.param().set("k_Cu", "400[W/(m*K)]", "copper thermal conductivity");
        model.param().set("rho_Cu", "8700[kg/m^3]", "copper density");
        model.param().set("Cp_Cu", "385[J/(kg*K)]", "copper heat capacity");
        model.param().set("E_Cu", "110[GPa]", "copper Young modulus");
        model.param().set("nu_Cu", "0.35", "copper Poisson ratio");
        model.param().set("alpha_Cu", "17e-6[1/K]", "copper thermal expansion");
        model.param().set("sigma_SAC", "7.1e6[S/m]", "solder conductivity");
        model.param().set("k_SAC", "58[W/(m*K)]", "solder thermal conductivity");
        model.param().set("rho_SAC", "7400[kg/m^3]", "solder density");
        model.param().set("Cp_SAC", "230[J/(kg*K)]", "solder heat capacity");
        model.param().set("E_SAC", "40[GPa]", "solder Young modulus");
        model.param().set("nu_SAC", "0.35", "solder Poisson ratio");
        model.param().set("alpha_SAC", "21e-6[1/K]", "solder thermal expansion");
        model.param().set("sigma_UF", "1e-10[S/m]", "underfill conductivity");
        model.param().set("k_UF", "0.6[W/(m*K)]", "underfill thermal conductivity");
        model.param().set("rho_UF", "1600[kg/m^3]", "underfill density");
        model.param().set("Cp_UF", "900[J/(kg*K)]", "underfill heat capacity");
        model.param().set("E_UF", "8[GPa]", "underfill Young modulus");
        model.param().set("nu_UF", "0.3", "underfill Poisson ratio");
        model.param().set("alpha_UF", "30e-6[1/K]", "underfill thermal expansion");
        model.param().set("sigma_FR4", "1e-10[S/m]", "board conductivity");
        model.param().set("k_FR4", "0.3[W/(m*K)]", "board thermal conductivity");
        model.param().set("rho_FR4", "1900[kg/m^3]", "board density");
        model.param().set("Cp_FR4", "1369[J/(kg*K)]", "board heat capacity");
        model.param().set("E_FR4", "20[GPa]", "board Young modulus");
        model.param().set("nu_FR4", "0.15", "board Poisson ratio");
        model.param().set("alpha_FR4", "17e-6[1/K]", "board thermal expansion");
        model.param().set("sigma_Si", "1e-3[S/m]", "silicon conductivity");
        model.param().set("k_Si", "130[W/(m*K)]", "silicon thermal conductivity");
        model.param().set("rho_Si", "2329[kg/m^3]", "silicon density");
        model.param().set("Cp_Si", "700[J/(kg*K)]", "silicon heat capacity");
        model.param().set("E_Si", "130[GPa]", "silicon Young modulus");
        model.param().set("nu_Si", "0.28", "silicon Poisson ratio");
        model.param().set("alpha_Si", "2.6e-6[1/K]", "silicon thermal expansion");
        // Mesh sizes, as lengths (the mesh is in the geometry's own unit).
        model.param().set("mh_bulk", "0.8[mm]", "board mesh size");
        model.param().set("mh_pkg", "0.3[mm]", "package mesh size");
        model.param().set("mh_die", "0.5[mm]", "die mesh size");
        model.param().set("mh_bump", "0.12[mm]", "bump mesh size");

        String comp = "comp1";
        model.component().create(comp, true);

        GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
        g3.lengthUnit("mm");

        // Board, with its top face at z = 0.
        g3.create("blk1", "Block");
        g3.feature("blk1").label("Board");
        g3.feature("blk1").set("size", new String[]{"Lb", "Lb", "Tb"});
        g3.feature("blk1").set("pos", new String[]{"-Lb/2", "-Lb/2", "-Tb"});
        g3.feature("blk1").set("selresult", true);
        g3.run("blk1");

        // Substrate copper layer.
        g3.create("blk2", "Block");
        g3.feature("blk2").label("Substrate Copper");
        g3.feature("blk2").set("size", new String[]{"Ls", "Ls", "Ts"});
        g3.feature("blk2").set("pos", new String[]{"-Ls/2", "-Ls/2", "0"});
        g3.feature("blk2").set("selresult", true);
        g3.run("blk2");

        // Underfill, and the bump array subtracted from it (kept, so the
        // bumps stay domains of their own).
        g3.create("blk3", "Block");
        g3.feature("blk3").label("Underfill");
        g3.feature("blk3").set("size", new String[]{"Lu", "Lu", "Tu"});
        g3.feature("blk3").set("pos", new String[]{"-Lu/2", "-Lu/2", "Ts"});
        g3.feature("blk3").set("selresult", true);
        g3.run("blk3");

        // The array is centred on the package: its copies sit at the pitch's
        // multiples from the first bump, so the first one is offset by half
        // the array's own width.
        g3.create("cyl1", "Cylinder");
        g3.feature("cyl1").label("Solder Bump");
        g3.feature("cyl1").set("r", "rb");
        g3.feature("cyl1").set("h", "Tu");
        g3.feature("cyl1").set("pos", new String[]{"-(nb-1)*pitch/2", "-(nb-1)*pitch/2", "Ts"});
        g3.feature("cyl1").set("axis", new double[]{0, 0, 1});
        g3.feature("cyl1").set("selresult", true);
        g3.run("cyl1");

        g3.create("arr1", "Array");
        g3.feature("arr1").selection("input").set("cyl1");
        g3.feature("arr1").set("fullsize", new String[]{"nb", "nb", "1"});
        g3.feature("arr1").set("displ", new String[]{"pitch", "pitch", "0"});
        g3.selection().create("csel1", "CumulativeSelection");
        g3.selection("csel1").label("Solder Bumps");
        g3.feature("arr1").set("contributeto", "csel1");
        g3.run("arr1");

        g3.create("dif1", "Difference");
        g3.feature("dif1").selection("input").set("blk3");
        g3.feature("dif1").selection("input2").named("csel1");
        g3.feature("dif1").set("keep", true);
        g3.run("dif1");

        // Die metal and the die itself, centred on the package.
        g3.create("blk4", "Block");
        g3.feature("blk4").label("Die Metal");
        g3.feature("blk4").set("size", new String[]{"Lu", "Lu", "Tm"});
        g3.feature("blk4").set("pos", new String[]{"-Lu/2", "-Lu/2", "Ts+Tu"});
        g3.feature("blk4").set("selresult", true);
        g3.run("blk4");

        g3.create("blk5", "Block");
        g3.feature("blk5").label("Die");
        g3.feature("blk5").set("size", new String[]{"Ld", "Ld", "Td"});
        g3.feature("blk5").set("pos", new String[]{"-Ld/2", "-Ld/2", "Ts+Tu+Tm"});
        g3.feature("blk5").set("selresult", true);
        g3.run("blk5");

        g3.run("fin");
        System.out.println("GEOM_OK");

        // ---- Domain identification ----
        GeomInfo gi = model.component(comp).geom("geom1");
        final int nDom = gi.getNDomains();
        final int nFace = gi.getNFaces();
        final int nBump = (int) Math.round(model.param().evaluate("nb"));
        int[] board = entities(model, comp, "geom1_blk1_dom");
        int[] substrate = entities(model, comp, "geom1_blk2_dom");
        int[] bumps = entities(model, comp, "geom1_csel1_dom");
        int[] dieMetal = entities(model, comp, "geom1_blk4_dom");
        int[] die = entities(model, comp, "geom1_blk5_dom");
        // The block the bumps were subtracted from keeps a selection over its
        // result *and* the objects it kept: the underfill is the one domain of
        // it that is not a bump.
        int[] underfillBlock = entities(model, comp, "geom1_blk3_dom");
        int underfill = 0;
        for (int d : underfillBlock) {
            boolean isBump = false;
            for (int b : bumps)
                if (b == d) isBump = true;
            if (isBump) continue;
            if (underfill != 0)
                throw new IllegalStateException("two underfill domains: " + underfill + ", " + d);
            underfill = d;
        }
        // One domain each for the board, the substrate copper, the underfill,
        // the die metal and the die, plus one per bump.
        if (board.length != 1 || substrate.length != 1 || bumps.length != nBump * nBump
            || dieMetal.length != 1 || die.length != 1 || underfill == 0
            || nDom != 5 + nBump * nBump)
            throw new IllegalStateException("unexpected geometry classification: nDom=" + nDom
                + " board=" + board.length + " substrate=" + substrate.length
                + " bumps=" + bumps.length + " underfill=" + underfill
                + " dieMetal=" + dieMetal.length + " die=" + die.length);
        System.out.println("DOMAINS=" + nDom + " board=" + board[0] + " substrate=" + substrate[0]
            + " underfill=" + underfill + " bumps=" + bumps.length + " dieMetal=" + dieMetal[0]
            + " die=" + die[0]);

        // ---- Boundary identification ----
        // The three z levels a boundary condition sits at, in the geometry's
        // own unit: the board's bottom, the top of the substrate copper (the
        // ring the underfill leaves exposed) and the top of the die metal (the
        // ring the die leaves exposed).
        final double mm = model.param().evaluate("1[mm]");
        final double zBoard = model.param().evaluate("-Tb") / mm;
        final double zSubstrate = model.param().evaluate("Ts") / mm;
        final double zDieMetal = model.param().evaluate("Ts+Tu+Tm") / mm;
        final double sideSubstrate = model.param().evaluate("Ls/2") / mm;
        final double sideDieMetal = model.param().evaluate("Lu/2") / mm;
        final double tol = 1e-6;

        final int[][] updown = gi.getUpDown();
        if (updown[0].length < nFace || updown[1].length < nFace)
            throw new IllegalStateException("getUpDown() returned too few entries");
        int[] owner = new int[nFace + 1];
        boolean[] exterior = new boolean[nFace + 1];
        int nExterior = 0;
        for (int f = 1; f <= nFace; f++) {
            exterior[f] = updown[0][f - 1] == 0 || updown[1][f - 1] == 0;
            owner[f] = updown[1][f - 1] != 0 ? updown[1][f - 1] : updown[0][f - 1];
            if (exterior[f]) nExterior++;
        }
        System.out.println("FACES=" + nFace + " EXTERIOR=" + nExterior);

        java.util.List<Integer> terminal = new java.util.ArrayList<Integer>();
        java.util.List<Integer> ground = new java.util.ArrayList<Integer>();
        java.util.List<Integer> mount = new java.util.ArrayList<Integer>();
        for (int f = 1; f <= nFace; f++) {
            if (!exterior[f]) continue;
            double[][] points = sampleFace(gi, f, 3);
            if (points.length == 0)
                throw new IllegalStateException("exterior face " + f + " yielded no points");
            if (owner[f] == dieMetal[0] && allAt(points, 2, zDieMetal, tol))
                terminal.add(f);
            else if (owner[f] == substrate[0] && allAt(points, 2, zSubstrate, tol))
                ground.add(f);
            else if (owner[f] == board[0] && allAt(points, 2, zBoard, tol))
                mount.add(f);
        }
        // The terminal and the ground are rings around the face the die, and
        // the underfill, leave exposed: whatever way COMSOL splits such a ring
        // into faces, the selected set has to reach its four outer sides, or
        // the current would enter the layer over part of its edge only. The
        // ring of the die metal is `Lu` wide and the die `Ld`, so the exposed
        // band is (Lu-Ld)/2 = 0.5 mm; the substrate's is (Ls-Lu)/2 = 1 mm.
        if (terminal.isEmpty() || ground.isEmpty() || mount.isEmpty()
            || !reachesSides(terminal, gi, sideDieMetal, tol)
            || !reachesSides(ground, gi, sideSubstrate, tol))
            throw new IllegalStateException("boundary identification failed: terminal="
                + terminal + " ground=" + ground + " mount=" + mount.size());

        // The package is cooled by natural convection alone, on every exterior
        // face the current does not enter on. The board's underside is one of
        // them: a board in still air convects from both of its faces.
        java.util.List<Integer> convection = new java.util.ArrayList<Integer>();
        for (int f = 1; f <= nFace; f++) {
            if (!exterior[f]) continue;
            if (terminal.contains(f) || ground.contains(f)) continue;
            convection.add(f);
        }
        System.out.println("TERMINAL=" + terminal + " GROUND=" + ground + " MOUNT="
            + mount.size() + " CONV=" + convection.size() + " of " + nExterior);

        // ---- Materials ----
        setMaterial(model, comp, "mat_FR4", "FR4 (Circuit Board)", board,
            "sigma_FR4", "k_FR4", "rho_FR4", "Cp_FR4", "alpha_FR4", "E_FR4", "nu_FR4");
        int[] copper = concat(substrate, dieMetal);
        setMaterial(model, comp, "mat_Cu", "Copper", copper,
            "sigma_Cu", "k_Cu", "rho_Cu", "Cp_Cu", "alpha_Cu", "E_Cu", "nu_Cu");
        setMaterial(model, comp, "mat_UF", "Underfill", new int[]{underfill},
            "sigma_UF", "k_UF", "rho_UF", "Cp_UF", "alpha_UF", "E_UF", "nu_UF");
        setMaterial(model, comp, "mat_SAC", "SAC305 Solder", bumps,
            "sigma_SAC", "k_SAC", "rho_SAC", "Cp_SAC", "alpha_SAC", "E_SAC", "nu_SAC");
        setMaterial(model, comp, "mat_Si", "Silicon", die,
            "sigma_Si", "k_Si", "rho_Si", "Cp_Si", "alpha_Si", "E_Si", "nu_Si");
        System.out.println("MAT_OK");

        // ---- Physics ----
        model.component(comp).physics().create("ec", "ConductiveMedia", "geom1");
        model.component(comp).physics().create("ht", "HeatTransfer", "geom1");
        model.component(comp).physics().create("solid", "SolidMechanics", "geom1");

        model.component(comp).physics("ec").create("term1", "Terminal", 2);
        model.component(comp).physics("ec").feature("term1").selection().set(ints(terminal));
        model.component(comp).physics("ec").feature("term1").set("TerminalType", "Voltage");
        model.component(comp).physics("ec").feature("term1").set("V0", "Vtot");
        model.component(comp).physics("ec").create("gnd1", "Ground", 2);
        model.component(comp).physics("ec").feature("gnd1").selection().set(ints(ground));

        model.component(comp).physics("ht").feature("init1").set("Tinit_src", "userdef");
        model.component(comp).physics("ht").feature("init1").set("Tinit", "T0");
        model.component(comp).physics("ht").create("hs1", "HeatSource", 3);
        model.component(comp).physics("ht").feature("hs1").selection().set(die);
        model.component(comp).physics("ht").feature("hs1").set("Q0", "q_chip");
        model.component(comp).physics("ht").create("hf1", "HeatFluxBoundary", 2);
        model.component(comp).physics("ht").feature("hf1").selection().set(ints(convection));
        model.component(comp).physics("ht").feature("hf1").set("HeatFluxType", "ConvectiveHeatFlux");
        model.component(comp).physics("ht").feature("hf1")
            .set("HeatTransferCoefficientType", "UserDef");
        model.component(comp).physics("ht").feature("hf1").set("h", "hconv");
        model.component(comp).physics("ht").feature("hf1").set("Text", "T0");

        // The board's underside is the mechanical support: the package is held
        // there, and nothing else constrains its rigid-body motion. It is a
        // convection face like the rest — a board in still air loses heat from
        // both of its faces — so the mechanical support is the only thing this
        // face carries.
        model.component(comp).physics("solid").feature("lemm1").set("E_mat", "from_mat");
        model.component(comp).physics("solid").feature("lemm1").set("nu_mat", "from_mat");
        model.component(comp).physics("solid").create("fix1", "Fixed", 2);
        model.component(comp).physics("solid").feature("fix1").selection().set(ints(mount));

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
        // Only the maximum element size is stated: COMSOL derives the minimum
        // from it, so the mesh stays coarse in the board and follows the bump
        // radius where the geometry asks for it.
        com.comsol.model.MeshFeature global = model.component(comp).mesh("mesh1").feature("size");
        global.set("custom", "on");
        global.set("hmax", "mh_bulk");
        global.set("hgrad", 1.4);
        global.set("hcurve", 0.3);

        com.comsol.model.MeshFeature sizePkg = model.component(comp).mesh("mesh1")
            .create("size1", "Size");
        sizePkg.selection().geom("geom1", 3);
        sizePkg.selection().set(concat(concat(substrate, new int[]{underfill}), dieMetal));
        sizePkg.set("custom", "on");
        sizePkg.set("hmax", "mh_pkg");

        com.comsol.model.MeshFeature sizeDie = model.component(comp).mesh("mesh1")
            .create("size2", "Size");
        sizeDie.selection().geom("geom1", 3);
        sizeDie.selection().set(die);
        sizeDie.set("custom", "on");
        sizeDie.set("hmax", "mh_die");

        com.comsol.model.MeshFeature sizeBump = model.component(comp).mesh("mesh1")
            .create("size3", "Size");
        sizeBump.selection().geom("geom1", 3);
        sizeBump.selection().set(bumps);
        sizeBump.set("custom", "on");
        sizeBump.set("hmax", "mh_bump");

        model.component(comp).mesh("mesh1").create("ftet1", "FreeTet");
        model.component(comp).mesh("mesh1").run();
        com.comsol.model.MeshSequence mesh = model.component(comp).mesh("mesh1");
        System.out.println("MESH_OK elements=" + mesh.getNumElem()
            + " vertices=" + mesh.getNumVertex());

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
        System.out.println("AdvancedPackageTransient_OK");
    }

    /// The entities of a geometry selection the sequence created, by the name
    /// the component exposes it under.
    private static int[] entities(Model model, String comp, String tag) {
        Selection s = model.component(comp).selection(tag);
        return s.entities();
    }

    /// Whether every point of `points` has the coordinate `k` equal to `value`.
    private static boolean allAt(double[][] points, int k, double value, double tol) {
        for (double[] p : points)
            if (Math.abs(p[k] - value) > tol) return false;
        return true;
    }

    /// Whether the sampled points of the faces in `faces` reach all four
    /// outer sides of the square ring of half width `side`: a set covering
    /// only part of the ring's edge would leave the current entering the
    /// layer over that part alone.
    private static boolean reachesSides(java.util.List<Integer> faces, GeomInfo gi,
        double side, double tol) {
        double xMin = Double.MAX_VALUE, xMax = -Double.MAX_VALUE;
        double yMin = Double.MAX_VALUE, yMax = -Double.MAX_VALUE;
        for (int f : faces)
            for (double[] p : sampleFace(gi, f, 3)) {
                xMin = Math.min(xMin, p[0]);
                xMax = Math.max(xMax, p[0]);
                yMin = Math.min(yMin, p[1]);
                yMax = Math.max(yMax, p[1]);
            }
        return xMin <= -side + tol && xMax >= side - tol
            && yMin <= -side + tol && yMax >= side - tol;
    }

    /// An (n+1) x (n+1) grid of points on face `f`, in its parameter space.
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

    private static int[] concat(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static int[] ints(java.util.List<Integer> list) {
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) out[i] = list.get(i);
        return out;
    }

    /// Write one material on `doms`, with E and nu in the Enu material model.
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
