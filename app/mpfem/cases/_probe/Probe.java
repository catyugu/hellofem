import com.comsol.model.GeomInfo;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/**
 * Probe v5.
 * Key facts learned:
 *  - comsolbatch invokes main(null) — never dereference args unguarded.
 *  - comsolcompile emits NO inner-class files; keep main flat (no anonymous
 *    inner classes / lambdas).
 *  - model.save(path, "java") writes Save-As-Java. On a case-insensitive FS the
 *    output must NOT collide with this class's source name (Probe.java), else it
 *    clobbers the source — generated_model.java is used instead.
 *  - The mesh text export is a RESULTS-node export feature, not the mesh-sequence
 *    export: result().export().create("mesh1","Mesh") + props. (MeshExport on the
 *    mesh sequence only knows "stlformat".)
 */
public class Probe {

    public static void main(String[] args) {
        String[] a = args == null ? new String[0] : args;
        final String[] P = a.length >= 4 ? a : new String[]{"result.txt", "mesh.mphtxt", "probe.mph", "generated_model.java"};

        try {
            Model model = ModelUtil.create("Model");
            model.param().set("L", "0.1[m]", "length");
            model.param().set("V0", "1[V]", "applied voltage");
            model.param().set("sig", "5.8e7[S/m]", "conductivity");

            String comp = "comp1";
            model.component().create(comp, true);

            com.comsol.model.GeomSequence g3 = model.component(comp).geom().create("geom1", 3);
            g3.create("blk1", "Block");
            g3.feature("blk1").set("size", new String[]{"L", "L", "L"});
            g3.feature("blk1").set("pos", new String[]{"0", "0", "0"});
            g3.run();
            System.out.println("GEOM_OK");

            GeomInfo gi = model.component(comp).geom("geom1");
            System.out.println("NDOM=" + gi.getNDomains() + " NFACE=" + gi.getNFaces());

            try {
                model.component(comp).material().create("mat1", "Common");
                model.component(comp).material("mat1").selection().set(new int[]{1});
                model.component(comp).material("mat1").propertyGroup("def")
                        .set("electricconductivity", new String[][]{{"sig"}});
                System.out.println("MAT_OK");
            } catch (Throwable t) {
                System.out.println("MAT_EXC: " + t);
            }

            try {
                model.component(comp).physics().create("ec", "ConductiveMedia", "geom1");
                model.component(comp).physics("ec").create("term1", "Terminal", 2);
                model.component(comp).physics("ec").feature("term1").selection().set(new int[]{1});
                model.component(comp).physics("ec").feature("term1").set("TerminalType", "Voltage");
                model.component(comp).physics("ec").feature("term1").set("V0", "V0");
                model.component(comp).physics("ec").create("gnd1", "Ground", 2);
                model.component(comp).physics("ec").feature("gnd1").selection().set(new int[]{2});
                System.out.println("PHYS_OK");
            } catch (Throwable t) {
                System.out.println("PHYS_EXC: " + t);
            }

            try {
                model.component(comp).mesh().create("mesh1");
                model.component(comp).mesh("mesh1").run();
                System.out.println("MESH_OK");
            } catch (Throwable t) {
                System.out.println("MESH_EXC: " + t);
            }

            try {
                model.study().create("std1");
                model.study("std1").create("stat", "Stationary");
                model.study("std1").createAutoSequences("stat");
                model.study("std1").run();
                System.out.println("STUDY_OK");
            } catch (Throwable t) {
                System.out.println("STUDY_EXC: " + t);
            }

            try {
                model.result().export().create("data1", "Data");
                model.result().export("data1").set("data", "dset1");
                model.result().export("data1").set("filename", P[0]);
                model.result().export("data1").set("expr", new String[]{"V", "ec.normJ"});
                model.result().export("data1").run();
                System.out.println("RESEXP_OK");
            } catch (Throwable t) {
                System.out.println("RESEXP_EXC: " + t);
            }

            // Mesh text export = RESULTS-node export feature "Mesh" (props differ by format)
            try {
                model.result().export().create("mesh1", "Mesh");
                model.result().export("mesh1").set("data", "dset1");
                model.result().export("mesh1").set("filename", P[1]);
                model.result().export("mesh1").run();
                System.out.println("MESHEXP_OK");
            } catch (Throwable t) {
                System.out.println("MESHEXP_EXC: " + t);
            }

            try {
                model.save(P[2]);
                System.out.println("SAVE_OK");
            } catch (Throwable t) {
                System.out.println("SAVE_EXC: " + t);
            }

            try {
                model.save(P[3], "java");
                System.out.println("SAVEJAVA_OK");
            } catch (Throwable t) {
                System.out.println("SAVEJAVA_EXC: " + t);
            }

            System.out.println("PROBE_DONE");
        } catch (Throwable t) {
            System.out.println("MAIN_EXC: " + t);
        }
    }
}
