"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
    Zap,
    Cpu,
    Layers,
    Settings,
    LayoutDashboard,
    ShieldCheck,
    LibraryBig,
    BarChart3
} from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

const navItems = [
    { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
    { name: "Generate", href: "/dashboard/generate", icon: Zap },
    { name: "Placement", href: "/dashboard/placement", icon: Cpu },
    { name: "DFM Analysis", href: "/dashboard/dfm", icon: ShieldCheck },
    { name: "Templates", href: "/dashboard/templates", icon: LibraryBig },
    { name: "Architecture", href: "/dashboard/architecture", icon: BarChart3 },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className="w-56 flex flex-col bg-dark-800/90 border-r border-border-subtle flex-shrink-0 z-20 backdrop-blur-sm">
            {/* Brand */}
            <div className="h-14 flex items-center px-5 border-b border-border-subtle">
                <Link href="/" className="flex items-center gap-2 group">
                    <div className="w-7 h-7 rounded bg-primary/20 flex items-center justify-center border border-primary/30 group-hover:border-primary transition-colors">
                        <Cpu className="w-4 h-4 text-primary" />
                    </div>
                    <span className="font-display font-bold text-base tracking-tight text-white">Circuit<span className="text-primary">Mind</span></span>
                </Link>
            </div>

            {/* Nav */}
            <nav className="flex-1 py-4 px-3 space-y-0.5">
                {navItems.map((item) => {
                    const isActive = pathname === item.href;
                    return (
                        <Link
                            key={item.name}
                            href={item.href}
                            className={cn(
                                "group flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 relative text-sm",
                                isActive
                                    ? "text-primary bg-primary/8 font-medium"
                                    : "text-white/40 hover:text-white/80 hover:bg-white/4"
                            )}
                        >
                            {isActive && (
                                <motion.div
                                    layoutId="sidebar-active"
                                    className="absolute left-0 w-1 h-6 bg-primary rounded-r-full"
                                />
                            )}
                            <item.icon className={cn("w-5 h-5 transition-colors", isActive ? "text-primary" : "group-hover:text-white")} />
                            <span className="text-sm font-display tracking-wide">{item.name}</span>
                        </Link>
                    );
                })}
            </nav>

            {/* Footer / Settings */}
            <div className="p-4 border-t border-border-subtle">
                <button className="flex items-center gap-3 px-3 py-2 w-full text-white/50 hover:text-white transition-colors rounded-md group">
                    <Settings className="w-5 h-5 group-hover:rotate-45 transition-transform duration-500" />
                    <span className="text-sm font-display">System Settings</span>
                </button>
                <div className="mt-4 px-3 py-3 rounded-lg bg-dark-900/50 border border-border-subtle">
                    <div className="flex items-center justify-between text-[10px] text-white/40 uppercase tracking-widest mb-2">
                        <span>Core Version</span>
                        <span className="text-primary">v2.4.1</span>
                    </div>
                    <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: "75%" }}
                            transition={{ duration: 1.5, ease: "easeOut" }}
                            className="h-full bg-primary/40"
                        />
                    </div>
                </div>
            </div>
        </aside>
    );
}
