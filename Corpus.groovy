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
  new File("data/go-plus-el.owl"))
OWLDataFactory dataFactory = manager.getOWLDataFactory()
ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor()
OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor)
ElkReasonerFactory reasonerFactory = new ElkReasonerFactory()
//OWLReasoner reasoner = reasonerFactory.createReasoner(ont, config)
//reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)

println(ont.getTBoxAxioms().size())
return

def renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl()
def shortFormProvider = new SimpleShortFormProvider()

def getName = { cl ->
  return shortFormProvider.getShortForm(cl);
}

out = new PrintWriter(
    new BufferedWriter(new FileWriter("data/output.txt")))

// def ignoredWords = ["\\r\\n|\\r|\\n", "[()]"]

ont.getClassesInSignature(true).each { cl ->
    clName = getName(cl)
    if (clName.startsWith("HP_")) {
	EntitySearcher.getEquivalentClasses(cl, ont).each { cExpr ->
	    if (!cExpr.isClassExpressionLiteral()) {
		String definition = renderer.render(cExpr);
		// ignoredWords.each { word ->
		//     definition = definition.replaceAll(word, "");
		// }
		out.println(clName + ": " + definition);
	    }
	}
    }
}

out.flush()
out.close()
