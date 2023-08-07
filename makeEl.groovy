@Grapes([
    @Grab(group="org.semanticweb.elk", module="elk-owlapi", version="0.4.3"),
    @Grab(group="net.sourceforge.owlapi", module="owlapi-api", version="4.2.5"),
    @Grab(group="net.sourceforge.owlapi", module="owlapi-apibinding", version="4.2.5"),
    @Grab(group="net.sourceforge.owlapi", module="owlapi-impl", version="4.2.5"),
    @Grab(group="net.sourceforge.owlapi", module="owlapi-parsers", version="4.2.5"),
    @Grab(group="de.tu-dresden.inf.lat.jcel", module="jcel-owlapi", version="0.24.1"),
    @Grab(group='commons-cli', module='commons-cli', version='1.4'),
    @GrabConfig(systemClassLoader=true)
])

import java.util.logging.Logger
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.*
import org.semanticweb.owlapi.reasoner.*
import de.tudresden.inf.lat.jcel.owlapi.main.*
import org.semanticweb.owlapi.profiles.*

def diri = new File(args[0])

OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ont = manager.loadOntologyFromOntologyDocument(diri)
OWLDataFactory fac = manager.getOWLDataFactory()

File file = new File(args[1])
OWLOntology ont2 = manager.createOntology()

OWL2ELProfile prof = new OWL2ELProfile()

def report = prof.checkOntology(ont)
def viol = report.getViolations()
def ignoreSet = new TreeSet()
viol.each { 
  if (it.getAxiom()!=null) {
    ignoreSet.add(it.getAxiom())
  }
}


def s = ont.getAxioms()
s.each {
  if (
    (! ignoreSet.contains(it))
  ) {
    manager.addAxiom(ont2,it)
  }
}

// ont2.getAxioms(AxiomType.ANNOTATION_ASSERTION).each {
//   println it
// }

manager.saveOntology(ont2, IRI.create(file.toURI()))
