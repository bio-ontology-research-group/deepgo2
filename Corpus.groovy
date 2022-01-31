@Grapes([
  @Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.2'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.1.0'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.1.0'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.1.0'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-parsers', version='4.1.0'),
  @GrabConfig(systemClassLoader=true)
  ])

import org.semanticweb.owlapi.model.parameters.*
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.*
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.owllink.*;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.search.*;
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.elk.owlapi.ElkReasonerConfiguration
import org.semanticweb.elk.reasoner.config.*
import org.semanticweb.owlapi.manchestersyntax.renderer.*
import org.semanticweb.owlapi.formats.*
import java.util.*


OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ont = manager.loadOntologyFromOntologyDocument(
  new File("data/go.owl"))
OWLDataFactory dataFactory = manager.getOWLDataFactory()
ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor()
OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor)
ElkReasonerFactory reasonerFactory = new ElkReasonerFactory()
OWLReasoner reasoner = reasonerFactory.createReasoner(ont, config)
reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)

print(ont.getAxioms().size())
return

def renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl()
def shortFormProvider = new SimpleShortFormProvider()

def getName = { cl ->
  return shortFormProvider.getShortForm(cl);
}

// def out = new PrintWriter(
//     new BufferedWriter(new FileWriter("data/swissprot_annots_new.tab")))

// def progress = 0;
// new File("data/swissprot_annots.tab").splitEachLine('\t') { items ->
//     def prot_id = items[0];
//     def gos = new LinkedList<OWLClass>();
//     def used = new HashSet<OWLClass>();
//     for (int i = 1; i < items.size(); i++) {
//         def goClass = dataFactory.getOWLClass(IRI.create(
//                 "http://purl.obolibrary.org/obo/" + items[i]));
//         gos.add(goClass);
//         used.add(goClass);
//     }
//     def goSet = new HashSet<OWLClass>();
//     while(!gos.isEmpty()) {
//         def go_id = gos.poll();
//         def name = getName(go_id);
// 	goSet.add(go_id);
//         if (name.startsWith("DG_")) {
//             continue;
//         }
// 	def eqClasses = EntitySearcher.getEquivalentClasses(go_id, ont);
//         if (eqClasses.size() > 0) {
//             eqClasses.each { expr ->
//                 if (expr.getClassExpressionType() == ClassExpressionType.valueOf("OWL_CLASS")) {
//                     def goClass = (OWLClass)expr;
//                     if (!used.contains(goClass)) {
//                         gos.add(goClass);
//                         used.add(goClass);
//                     }
//                 } else if (expr.getClassExpressionType() == ClassExpressionType.valueOf("OBJECT_INTERSECTION_OF")) {
//                     expr.asConjunctSet().each { cexpr ->
//                         def goClass = (OWLClass)cexpr;
//                         if (!used.contains(goClass)) {
//                             gos.add(goClass);
//                             used.add(goClass);
//                         }
//                     }
//                 }
//             };
//         }  
        
//     }
//     def superGos = new HashSet<OWLClass>(goSet);
//     for (OWLClass go: goSet) {
//         reasoner.getSuperClasses(go, false).each { node ->
//             superGos.addAll(node.getEntities());
//         }
//     }
//     out.print(prot_id);
//     for (OWLClass go: superGos) {
//         def name = getName(go);
//         if (name.startsWith("GO_") || name.startsWith("DG_")) {
//             out.print("\t" + name.replaceAll("_", ":"));
//         }
//     }
//     out.println();
//     progress++;
//     if (progress % 1000 == 0) {
//         System.out.println("Progress: " + progress);
//     }
// }

// out.flush()
// out.close()


out = new PrintWriter(
    new BufferedWriter(new FileWriter("data/definitions_go.txt")))

def ignoredWords = ["\\r\\n|\\r|\\n", "[()]"]

ont.getClassesInSignature(true).each { cl ->
    clName = getName(cl)
    if (clName.startsWith("GO_")) {
	EntitySearcher.getEquivalentClasses(cl, ont).each { cExpr ->
	    if (!cExpr.isClassExpressionLiteral()) {
		String definition = renderer.render(cExpr);
		ignoredWords.each { word ->
		    definition = definition.replaceAll(word, "");
		}
		out.println(clName + ": " + definition);
	    }
	}
    }
}

ont.getClassesInSignature(true).each { cl ->
    clName = getName(cl)
    if (clName.startsWith("DG_")) {
	EntitySearcher.getEquivalentClasses(cl, ont).each { cExpr ->
	    if (!cExpr.isClassExpressionLiteral()) {
		String definition = renderer.render(cExpr);
		ignoredWords.each { word ->
		    definition = definition.replaceAll(word, "");
		}
		out.println(clName + ": " + definition);
	    }
	}
    }
}

out.flush()
out.close()
