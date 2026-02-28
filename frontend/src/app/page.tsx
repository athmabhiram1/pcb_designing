"use client";

import React from "react";
import Link from "next/link";
import { motion, useScroll, useTransform } from "framer-motion";
import { Cpu, Zap, ShieldCheck, ArrowRight, Layers, BarChart3, ChevronRight } from "lucide-react";
import { MechButton } from "@/components/ui/MechButton";
import { NodeCanvas } from "@/components/ui/NodeCanvas";

export default function HeroPage() {
  const { scrollYProgress } = useScroll();
  const y1 = useTransform(scrollYProgress, [0, 1], [0, -200]);
  const y2 = useTransform(scrollYProgress, [0, 1], [0, -500]);

  return (
    <div className="relative min-h-screen bg-dark-900 overflow-hidden font-sans">
      <NodeCanvas />

      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 glass-nav">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded bg-primary/20 flex items-center justify-center border border-primary/30">
              <Cpu className="w-6 h-6 text-primary" />
            </div>
            <span className="font-display font-extrabold text-xl tracking-tight text-white uppercase">
              Circuit<span className="text-primary italic">Mind</span>
            </span>
          </div>

          <div className="hidden md:flex items-center gap-8 text-sm font-display tracking-widest text-white/60 uppercase">
            <Link href="#features" className="hover:text-primary transition-colors">Architecture</Link>
            <Link href="#workflow" className="hover:text-primary transition-colors">Workflow</Link>
            <Link href="#pricing" className="hover:text-primary transition-colors">Cloud Force</Link>
          </div>

          <Link href="/dashboard">
            <MechButton variant="primary" size="md">
              Launch Terminal <ChevronRight className="w-4 h-4 ml-1" />
            </MechButton>
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-40 pb-32 px-6">
        <div className="max-w-7xl mx-auto flex flex-col items-center text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 text-primary text-[10px] font-bold uppercase tracking-[0.3em] mb-8"
          >
            <Zap className="w-3 h-3" /> System Status: Operational v2.4
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-5xl md:text-8xl font-display font-black text-white leading-[0.9] tracking-tighter mb-8 max-w-5xl uppercase"
          >
            Expert-Grade <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-secondary to-primary animate-pulse-slow">
              Generative PCB
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-lg md:text-xl text-white/50 max-w-2xl font-light leading-relaxed mb-12"
          >
            Bridge the gap between concept and fabrication. Our AI backend generates production-ready netlists,
            optimal layouts, and DFM-verified schematics in seconds.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto"
          >
            <Link href="/dashboard/generate">
              <MechButton variant="primary" size="lg" className="w-full sm:w-auto">
                Start Generation <ArrowRight className="w-5 h-5 ml-2" />
              </MechButton>
            </Link>
            <MechButton variant="outline" size="lg" className="w-full sm:w-auto">
              View Specs
            </MechButton>
          </motion.div>
        </div>

        {/* Floating elements */}
        <motion.div style={{ y: y1 }} className="absolute -left-20 top-1/2 w-64 h-64 border border-white/5 rounded-full pointer-events-none" />
        <motion.div style={{ y: y2 }} className="absolute -right-40 top-1/3 w-96 h-96 border border-primary/10 rounded-full pointer-events-none" />
      </section>

      {/* Feature Grid (Bento) */}
      <section id="features" className="py-32 px-6 relative bg-dark-800/50">
        <div className="max-w-7xl mx-auto">
          <div className="bento-grid absolute inset-0 opacity-20" />

          <div className="grid grid-cols-1 md:grid-cols-4 md:grid-rows-2 gap-4 h-[800px] relative z-10">
            {/* Big Feature 1 */}
            <motion.div
              whileHover={{ scale: 1.01 }}
              className="md:col-span-2 md:row-span-2 p-8 rounded-2xl bg-dark-700 border border-border-subtle flex flex-col justify-end group overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-12 opacity-10 group-hover:scale-110 transition-transform duration-700">
                <Zap className="w-64 h-64 text-primary" />
              </div>
              <Zap className="w-12 h-12 text-primary mb-6" />
              <h3 className="text-3xl font-display font-bold text-white mb-4 uppercase">Sub-Millisecond Routing</h3>
              <p className="text-white/40 max-w-sm">
                Proprietary AI kernels handle complex BGA routing and multi-layer impedance matching with zero human intervention.
              </p>
            </motion.div>

            {/* Feature 2 */}
            <motion.div
              whileHover={{ scale: 1.02 }}
              className="md:col-span-2 p-8 rounded-2xl bg-dark-700 border border-border-subtle flex items-center justify-between group"
            >
              <div>
                <ShieldCheck className="w-10 h-10 text-secondary mb-4" />
                <h3 className="text-xl font-display font-bold text-white mb-2 uppercase">DFM Validation</h3>
                <p className="text-white/40 max-w-xs text-sm italic">Tested against JLCPCB, PCBWay, and Sierra Circuits rulesets.</p>
              </div>
              <div className="w-24 h-24 rounded bg-secondary/5 border border-secondary/20 flex items-center justify-center">
                <div className="w-12 h-12 border-2 border-dashed border-secondary/40 animate-spin-slow rounded" />
              </div>
            </motion.div>

            {/* Feature 3 */}
            <motion.div
              whileHover={{ scale: 1.02 }}
              className="p-8 rounded-2xl bg-dark-700 border border-border-subtle group"
            >
              <Layers className="w-8 h-8 text-primary/60 mb-4" />
              <h4 className="text-lg font-display font-bold text-white mb-2 uppercase">Multi-Layer</h4>
              <div className="flex flex-col gap-1">
                {[1, 2, 3, 4].map(l => <div key={l} className="h-1 bg-primary/20 rounded-full" style={{ width: `${100 - l * 15}%` }} />)}
              </div>
            </motion.div>

            {/* Feature 4 */}
            <motion.div
              whileHover={{ scale: 1.02 }}
              className="p-8 rounded-2xl bg-dark-700 border border-border-subtle group"
            >
              <BarChart3 className="w-8 h-8 text-accent mb-4" />
              <h4 className="text-lg font-display font-bold text-white mb-2 uppercase">Cost Realtime</h4>
              <p className="text-white/40 text-xs">Dynamic pricing injection based on current market component availability.</p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-20 border-t border-border-subtle px-6 bg-dark-900">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-12">
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <Cpu className="w-6 h-6 text-primary" />
              <span className="font-display font-black text-xl text-white uppercase tracking-tighter">CircuitMind</span>
            </div>
            <p className="text-white/30 text-sm max-w-xs">Industrial AI for next-generation hardware engineering.</p>
          </div>

          <div className="flex gap-12 text-sm font-mono text-white/40 uppercase tracking-widest">
            <div className="flex flex-col gap-2">
              <span className="text-white/20 mb-2">Protocol</span>
              <span>Security</span>
              <span>Latency</span>
            </div>
            <div className="flex flex-col gap-2">
              <span className="text-white/20 mb-2">Social</span>
              <span>GitHub</span>
              <span>X / Twitter</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
