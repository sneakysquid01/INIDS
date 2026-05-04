import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global_state.js';

// DOM references
const searchInput = document.getElementById('search-docs');
const categoryTabs = document.querySelectorAll('.category-tab');
const docsContainer = document.getElementById('docs-container');

// State
let currentCategory = 'all';
let searchQuery = '';

// Sample documentation data
const DOCS_DATA = [
    {
        id: 1,
        title: 'Getting Started with INIDS',
        category: 'getting-started',
        description: 'Learn the basics of INIDS 3.0 and set up your first detection engine.',
        content: 'INIDS (Intrusion Detection System) is a comprehensive network security monitoring platform. Start with the basics to understand how alerts, actions, and policies work together.',
        readTime: '5 min'
    },
    {
        id: 2,
        title: 'Installation & Setup',
        category: 'getting-started',
        description: 'Step-by-step guide to install and configure INIDS for your environment.',
        content: 'Follow this guide to deploy INIDS in your network. Learn about system requirements, dependencies, and initial configuration.',
        readTime: '10 min'
    },
    {
        id: 3,
        title: 'Configuring Detection Engines',
        category: 'configuration',
        description: 'Learn how to enable and configure detection engines (YARA, ML, Signature, Behavior).',
        content: 'Each detection engine serves a specific purpose. Configure them based on your security posture and threat model.',
        readTime: '8 min'
    },
    {
        id: 4,
        title: 'Policy Management',
        category: 'configuration',
        description: 'Create and manage security policies for automated response.',
        content: 'Policies define how INIDS responds to detected threats. Set up block rules, alert rules, and quarantine policies.',
        readTime: '7 min'
    },
    {
        id: 5,
        title: 'Understanding Alert Severity',
        category: 'detection',
        description: 'Learn about alert severity levels and how they impact your security operations.',
        content: 'INIDS uses severity levels (Critical, High, Medium, Low) to prioritize threats. Understand what each means.',
        readTime: '4 min'
    },
    {
        id: 6,
        title: 'Detection Workflow',
        category: 'detection',
        description: 'How INIDS analyzes traffic and generates alerts.',
        content: 'Learn the complete flow from packet capture to alert generation. Understand how multiple engines work together.',
        readTime: '6 min'
    },
    {
        id: 7,
        title: 'Troubleshooting Common Issues',
        category: 'troubleshooting',
        description: 'Solutions to frequently encountered problems.',
        content: 'If you encounter issues, check this guide for common problems and their solutions.',
        readTime: '12 min'
    },
    {
        id: 8,
        title: 'Performance Optimization',
        category: 'troubleshooting',
        description: 'Tips to improve INIDS performance and reduce false positives.',
        content: 'Optimize your INIDS deployment for better performance and accuracy.',
        readTime: '9 min'
    },
    {
        id: 9,
        title: 'REST API Reference',
        category: 'api',
        description: 'Complete REST API documentation for programmatic access.',
        content: 'Use the INIDS REST API to programmatically query alerts, actions, policies, and more.',
        readTime: '15 min'
    },
    {
        id: 10,
        title: 'WebSocket Events',
        category: 'api',
        description: 'Real-time event streaming via WebSocket.',
        content: 'Connect to real-time event streams for live monitoring and integration.',
        readTime: '8 min'
    }
];

/**
 * Format documentation card
 */
function formatDocCard(doc) {
    const card = document.createElement('div');
    card.className = 'bg-[#151922] border border-[#1a1f2e] rounded-lg p-4 hover:border-[#3b82f6] transition-colors cursor-pointer group';
    card.innerHTML = `
        <div class="flex items-start justify-between mb-2">
            <h3 class="text-white font-semibold text-sm group-hover:text-[#3b82f6] transition-colors">${doc.title}</h3>
            <span class="text-[#8f9099] text-xs bg-[#0a0c10] px-2 py-1 rounded">${doc.readTime}</span>
        </div>
        
        <p class="text-[#8f9099] text-xs mb-3 line-clamp-2">${doc.description}</p>
        
        <div class="text-[#6b7280] text-xs mb-3">${doc.content}</div>
        
        <div class="flex items-center justify-between">
            <span class="text-[#3b82f6] text-xs uppercase tracking-wider font-medium">
                ${doc.category.replace('-', ' ')}
            </span>
            <span class="text-[#3b82f6] group-hover:translate-x-1 transition-transform">→</span>
        </div>
    `;
    
    card.addEventListener('click', () => {
        AppToast.info(`Opened: ${doc.title}`);
    });
    
    return card;
}

/**
 * Filter docs
 */
function filterDocs(docs) {
    let filtered = docs;
    
    if (currentCategory !== 'all') {
        filtered = filtered.filter(d => d.category === currentCategory);
    }
    
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(d =>
            d.title?.toLowerCase().includes(q) ||
            d.description?.toLowerCase().includes(q) ||
            d.content?.toLowerCase().includes(q)
        );
    }
    
    return filtered;
}

/**
 * Render docs
 */
function renderDocs() {
    const filtered = filterDocs(DOCS_DATA);
    docsContainer.innerHTML = '';
    
    if (filtered.length === 0) {
        docsContainer.innerHTML = `
            <div class="text-center py-12">
                <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                    ${searchQuery ? 'No matching documentation found' : 'No docs in this category'}
                </div>
            </div>
        `;
        return;
    }
    
    filtered.forEach(doc => {
        try {
            const card = formatDocCard(doc);
            docsContainer.appendChild(card);
        } catch (err) {
            console.error('Error rendering doc:', err);
        }
    });
}

/**
 * Initialize page
 */
function initPage() {
    renderDocs();
    
    // Setup category tab handlers
    categoryTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            categoryTabs.forEach(t => {
                t.classList.remove('bg-[#3b82f6]', 'text-white');
                t.classList.add('text-[#8f9099]');
            });
            tab.classList.remove('text-[#8f9099]');
            tab.classList.add('bg-[#3b82f6]', 'text-white');
            
            currentCategory = tab.dataset.category;
            renderDocs();
        });
    });
    
    // Setup search
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderDocs();
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterDocs, renderDocs };
